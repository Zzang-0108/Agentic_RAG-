"""Offline tests. Never import the user's real rag.py or make model requests."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evaluate as ev


def sample_case():
    return {
        "id": "case-1", "question": "测试问题", "reference_answer": "测试答案",
        "category": "comparison", "split": "test", "answerable": True,
        "expected_need_retrieve": True,
        "reference_pages": [
            {"source": "knowledge_db/a.pdf", "page_number": 1},
            {"source": "knowledge_db/a.pdf", "page_number": 2},
        ],
    }


def doc(source="knowledge_db/a.pdf", page=1, text="证据"):
    return SimpleNamespace(metadata={"source": source, "page_number": page}, page_content=text)


class Retriever:
    k = 5

    def __init__(self, fail=False):
        self.fail = fail

    def invoke(self, question, config=None):
        if self.fail:
            raise RuntimeError("模拟接口失败")
        return [doc(), doc(page=2)]


class Agent:
    def invoke(self, state, config=None):
        return {"generation": "测试答案 [资料 1]", "documents": [doc(), doc(page=2)],
                "need_retrieve": True, "retries": 1}


def target():
    return SimpleNamespace(bm25_retriever=Retriever(), hnsw_retriever=Retriever(),
                           ensemble_retriever=Retriever(), rerank_retriever=Retriever(),
                           app=Agent())


class MetricsTests(unittest.TestCase):
    def test_cross_page_chunk_uses_actual_spans_without_inventing_intermediate_pages(self):
        item = doc()
        item.metadata.update(page_numbers=[1, 2, 3], source_spans=[
            {"page_number": 1, "start": 0, "end": 1}, {"page_number": 3, "start": 1, "end": 2}])
        serialized = ev.serialize_documents([item])
        self.assertEqual(serialized[0]["page_numbers"], [1, 3])
        json.dumps(serialized)  # 明细必须能保存为 JSON。
        result = ev.page_metrics(sample_case(), serialized, 5)
        self.assertEqual(result["page_recall_at_k"], .5)

    def test_plain_rate_limit_is_not_misclassified_as_balance_exhaustion(self):
        self.assertFalse(ev.resource_error({"message": "429 Too Many Requests"}))
        self.assertTrue(ev.resource_error({"provider_code": "1113", "message": "429"}))
        self.assertTrue(ev.resource_error({"message": 'Error: {"error":{"code":"1113"}}'}))
        self.assertFalse(ev.row_resource_error({"answer": "余额不足", "question": "1113"}))

    def test_http_response_error_code_is_preserved_without_response_dump(self):
        error = RuntimeError("429 Too Many Requests")
        error.response = SimpleNamespace(json=lambda: {"error": {"code": "1113", "message": "private"}})
        info = ev.error_info(error)
        self.assertEqual(info["provider_code"], "1113")
        self.assertNotIn("private", json.dumps(info))

    def test_retrieval_balance_failure_skips_judge(self):
        module = target()
        module.app = SimpleNamespace(invoke=lambda *a, **kw: {
            "generation": "本次检索未能完成", "need_retrieve": True, "documents": [],
            "evidence_status": "error", "trace": [{"node": "retrieve", "latency_ms": 1,
                "error": {"type": "APIReachLimitError", "message": "余额不足或无可用资源包"}}]})
        def judge(*args):
            raise AssertionError("余额耗尽后不得继续调用裁判")
        row = ev.evaluate_case(module, sample_case(), "agent", 5, judge=judge)
        self.assertEqual(row["judge_status"], "not_run_resource_error")
        self.assertTrue(ev.row_resource_error(row))
        self.assertNotIn("judgment", row)

    def test_three_way_routing_and_trace_survive_serialization(self):
        case = sample_case()
        case.update(category="clarification", answerable=False, expected_need_retrieve=False, reference_pages=[])
        module = target()
        module.app = SimpleNamespace(invoke=lambda *a, **kw: {
            "generation": "请提供比较对象", "need_retrieve": False, "retries": 0,
            "route_action": "clarify", "final_action": "clarify", "documents": [],
            "trace": [{"node": "route", "latency_ms": 12, "usage": {"llm_started_calls": 1}}]})
        row = ev.evaluate_case(module, case, "agent", 5)
        self.assertTrue(row["route_action_correct"])
        self.assertEqual(row["trace"][0]["node"], "route")
        self.assertEqual(ev.summarize([row])["agent"]["node_metrics"]["route"]["executions"], 1)

    def test_unanswerable_routing_policy_does_not_change_original_labels(self):
        case = sample_case()
        case.update(category="unanswerable", answerable=False, reference_pages=[])
        module = target()
        module.app = SimpleNamespace(invoke=lambda *a, **kw: {
            "generation": "请提供日志", "need_retrieve": False, "route_action": "clarify", "documents": []})
        row = ev.evaluate_case(module, case, "agent", 5)
        self.assertFalse(row["route_correct"])  # 原来的二分类标签仍保留
        self.assertIsNone(row["route_action_correct"])  # 不给策略歧义伪造金标

    def test_rerank_scores_are_retained(self):
        item = doc()
        item.metadata["rerank_score"] = .8
        self.assertEqual(ev.serialize_documents([item])[0]["rerank_score"], .8)

    def test_repeated_chunks_do_not_inflate_page_recall(self):
        docs = [doc(page=99), doc(), doc(), doc(page=2)]
        result = ev.page_metrics(sample_case(), ev.serialize_documents(docs), 3)
        self.assertEqual(result, {"page_hit_at_k": 1., "page_recall_at_k": .5, "mrr_at_k": .5})

    def test_same_page_number_in_another_book_does_not_match(self):
        result = ev.page_metrics(sample_case(), ev.serialize_documents([doc("other/a.pdf")]), 5)
        self.assertEqual(result["page_hit_at_k"], 0.)

    def test_absolute_and_relative_sources_match(self):
        source = str(ev.ROOT / "knowledge_db" / "a.pdf")
        result = ev.page_metrics(sample_case(), ev.serialize_documents([doc(source)]), 5)
        self.assertEqual(result["page_recall_at_k"], .5)

    def test_page_zero_metadata_is_converted_to_one(self):
        item = SimpleNamespace(metadata={"source": "a.pdf", "page": 0}, page_content="x")
        self.assertEqual(ev.serialize_documents([item])[0]["page_number"], 1)

    def test_no_gold_pages_is_not_a_zero_score(self):
        case = sample_case()
        case["reference_pages"] = []
        self.assertTrue(all(v is None for v in ev.page_metrics(case, [], 5).values()))

    def test_citation_identifier_validation(self):
        result = ev.citation_metrics("甲 [资料 1]，乙 [资料 3]", [{}, {}])
        self.assertEqual(result["citation_id_validity"], .5)
        self.assertEqual(result["invalid_citation_ids"], [3])
        self.assertIsNone(ev.citation_metrics("没有引用", [])['citation_id_validity'])

    def test_retrieval_failure_is_recorded_as_an_operational_miss(self):
        module = target()
        module.bm25_retriever = Retriever(fail=True)
        row = ev.evaluate_case(module, sample_case(), "bm25", 5)
        self.assertEqual(row["status"], "error")
        result = ev.summarize([row])["bm25"]
        self.assertEqual(result["error_rate"], 1)
        self.assertEqual(result["page_recall_at_k"], {"mean": 0., "n": 1})
        self.assertIsNone(result["latency_success_ms"]["p50"])

    def test_judge_failure_does_not_become_an_answer_score(self):
        def broken_judge(*args):
            raise ValueError("没有结构化结果")
        row = ev.evaluate_case(target(), sample_case(), "agent", 5, judge=broken_judge)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["judge_status"], "error")
        result = ev.summarize([row])["agent"]
        self.assertEqual(result["judge_failed"], 1)
        self.assertEqual(result["judge_answer_correct"], {"mean": None, "n": 0})

    def test_absent_route_is_unscored(self):
        module = target()
        module.app = SimpleNamespace(invoke=lambda *args, **kw: {"generation": "回答"})
        row = ev.evaluate_case(module, sample_case(), "agent", 5)
        self.assertIsNone(row["route_correct"])

    def test_empty_answer_is_an_execution_error(self):
        module = target()
        module.app = SimpleNamespace(invoke=lambda *args, **kw: {"generation": ""})
        self.assertEqual(ev.evaluate_case(module, sample_case(), "agent", 5)["status"], "error")

    def test_refusal_denominators_and_judge_coverage(self):
        def judge(case, row, tracker):
            return {"answer_correct": True, "grounded": None, "abstained": True,
                    "citations_supported": None, "reason": "模拟拒答"}
        case = sample_case()
        case.update(answerable=False, reference_pages=[])
        row = ev.evaluate_case(target(), case, "agent", 5, judge=judge)
        result = ev.summarize([row])["agent"]
        self.assertEqual(result["unanswerable_abstention_rate"], {"mean": 1., "n": 1})
        self.assertEqual(result["answerable_false_abstention_rate"], {"mean": None, "n": 0})
        self.assertEqual(result["judge_grounded"], {"mean": None, "n": 0})

    def test_percentile_interpolation(self):
        self.assertEqual(ev.percentile([10, 20], .5), 15)
        self.assertEqual(ev.percentile([10], .95), 10)
        self.assertIsNone(ev.percentile([], .95))


class IntegrationTests(unittest.TestCase):
    def prepare(self, root):
        (root / "knowledge_db").mkdir()
        (root / "knowledge_db/a.pdf").write_bytes(b"test file existence only")
        (root / "rag.py").write_text("raise RuntimeError('must not execute')", encoding="utf-8")
        data = root / "questions.jsonl"
        data.write_text(json.dumps(sample_case(), ensure_ascii=False) + "\n", encoding="utf-8")
        return data

    def test_validation_does_not_import_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.prepare(root)
            with patch.object(ev, "load_target", side_effect=AssertionError("unexpected import")):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = ev.main(["--dataset", str(data), "--module", str(root / "rag.py"), "--validate-only"])
            self.assertEqual(code, 0)

    def test_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.prepare(root)
            data.write_text(data.read_text(encoding="utf-8") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重复 id"):
                ev.load_dataset(data, root)

    def test_full_report_flow_uses_only_fake_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.prepare(root)
            output = root / "report"
            with patch.object(ev, "load_target", return_value=target()), patch.object(ev, "make_usage_tracker", return_value=None):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = ev.main(["--dataset", str(data), "--module", str(root / "rag.py"),
                                    "--mode", "both", "--output", str(output)])
            self.assertEqual(code, 0)
            rows = [json.loads(line) for line in (output / "details.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 5)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(set(summary["by_method"]), {"bm25", "dense", "hybrid", "rerank", "agent"})
            self.assertEqual(summary["by_method"]["agent"]["page_recall_at_k"]["mean"], 1.)
            self.assertTrue((output / "report.md").is_file())

    def test_balance_exhaustion_stops_before_next_method_or_question(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.prepare(root)
            cases = [sample_case(), dict(sample_case(), id="case-2")]
            data.write_text("\n".join(json.dumps(c) for c in cases), encoding="utf8")
            module = target()
            def fail(*args, **kwargs):
                raise RuntimeError('Error: {"error":{"code":"1113","message":"余额不足"}}')
            module.bm25_retriever = SimpleNamespace(k=5, invoke=fail)
            output = root / "report"
            with patch.object(ev, "load_target", return_value=module), patch.object(ev, "make_usage_tracker", return_value=None):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = ev.main(["--dataset", str(data), "--module", str(root / "rag.py"),
                                    "--mode", "both", "--output", str(output)])
            self.assertEqual(code, 3)
            summary = json.loads((output / "summary.json").read_text(encoding="utf8"))
            self.assertEqual(summary["status"], "resource_exhausted")
            self.assertEqual(summary["completed_evaluations"], 1)
            self.assertEqual(summary["planned_evaluations"], 10)
            self.assertEqual(len((output / "details.jsonl").read_text(encoding="utf8").splitlines()), 1)

    def test_progress_is_saved_before_next_call_and_interrupt_keeps_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.prepare(root)
            data.write_text("\n".join(json.dumps(c) for c in
                            [sample_case(), dict(sample_case(), id="case-2")]), encoding="utf8")
            output = root / "report"
            real_evaluate = ev.evaluate_case
            def evaluate(*args, **kwargs):
                if args[1]["id"] == "case-2":
                    checkpoint = json.loads((output / "summary.json").read_text(encoding="utf8"))
                    self.assertEqual(checkpoint["status"], "running")
                    self.assertEqual(checkpoint["completed_evaluations"], 1)
                    raise KeyboardInterrupt()
                return real_evaluate(*args, **kwargs)
            with patch.object(ev, "load_target", return_value=target()), patch.object(ev, "make_usage_tracker", return_value=None), patch.object(ev, "evaluate_case", side_effect=evaluate):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = ev.main(["--dataset", str(data), "--module", str(root / "rag.py"),
                                    "--mode", "agent", "--output", str(output)])
            self.assertEqual(code, 130)
            summary = json.loads((output / "summary.json").read_text(encoding="utf8"))
            self.assertEqual(summary["status"], "interrupted")
            self.assertEqual(summary["completed_evaluations"], 1)


# 直接用 make_judge 用的那一份 schema，不再手抄：手抄的副本会和真的漂移，而这一轮恰恰是
# schema 的校验行为（可选字段收到带引号的 "null"）决定了几款模型能不能用。
Judgment = ev.judgment_schema()


VALID = {"answer_correct": True, "grounded": True, "abstained": False,
         "citations_supported": None, "reason": "答案给出了留出法、交叉验证和自助法"}


class JudgeExtractionTests(unittest.TestCase):
    """从模型正文里抠 JSON。"""

    def test_fenced_json_with_envelope_is_recovered(self):
        # 实测 45 条失败里 44 条长这样：代码块 + {"Judgment": {...}} 信封
        content = "```json\n" + json.dumps({"Judgment": VALID}, ensure_ascii=False) + "\n```"
        parsed, note = ev.extract_structured_data(content, Judgment)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.reason, VALID["reason"])
        self.assertIn("来自正文", note)

    def test_bare_json_is_recovered(self):
        parsed, _ = ev.extract_structured_data(json.dumps(VALID, ensure_ascii=False), Judgment)
        self.assertIsNotNone(parsed)

    def test_fenced_json_without_a_language_tag_is_recovered(self):
        content = "```\n" + json.dumps(VALID) + "\n```"
        parsed, _ = ev.extract_structured_data(content, Judgment)
        self.assertIsNotNone(parsed)

    def test_json_embedded_in_prose_is_recovered(self):
        content = "我的判断如下：\n```json\n" + json.dumps(VALID) + "\n```\n以上。"
        parsed, _ = ev.extract_structured_data(content, Judgment)
        self.assertIsNotNone(parsed)

    def test_unknown_keys_are_dropped_instead_of_failing(self):
        payload = dict(VALID, confidence=0.9, notes="额外说明")
        parsed, note = ev.extract_structured_data(json.dumps(payload), Judgment)
        self.assertIsNotNone(parsed)
        self.assertIn("多余键", note)
        self.assertFalse(hasattr(parsed, "confidence"))

    def test_string_boolean_is_rejected(self):
        # StrictBool：模型把 true 写成字符串时必须判不通过，而不是悄悄当成 True
        parsed, note = ev.extract_structured_data(
            json.dumps(dict(VALID, answer_correct="true")), Judgment)
        self.assertIsNone(parsed)
        self.assertIn("不符合 schema", note)

    def test_the_optional_fields_accept_the_models_string_encoding(self):
        # glm-4-plus / glm-4-air / glm-4-airx 把可空的可选字段一律写成字符串：
        # null → "null"，true → "true"。不认这些写法，这三款模型在可回答题上
        # 只有 7/64 能解析出来（57 条卡在这里），--judge-model 等于摆设。
        for raw_value, expected in (("null", None), ("none", None), ("", None),
                                    ("true", True), ("false", False)):
            with self.subTest(raw_value=raw_value):
                payload = dict(VALID, grounded=raw_value, citations_supported=raw_value)
                parsed, _ = ev.extract_structured_data(json.dumps(payload), Judgment)
                self.assertIsNotNone(parsed)
                self.assertIs(parsed.grounded, expected)
                self.assertIs(parsed.citations_supported, expected)

    def test_the_required_fields_still_reject_string_booleans(self):
        # 宽容到这里为止。answer_correct / abstained 驱动全部核心指标，把 "true" coerce
        # 成 True 等于凭空造一个分数出来，比判不出来更糟。（glm-4-plus 在这两项上本来就
        # 输出真布尔值，所以这条严格并不会妨碍它。）
        for field in ("answer_correct", "abstained"):
            for raw_value in ("true", "false", "null"):
                with self.subTest(field=field, raw_value=raw_value):
                    parsed, _ = ev.extract_structured_data(
                        json.dumps(dict(VALID, **{field: raw_value})), Judgment)
                    self.assertIsNone(parsed)

    def test_unusable_content_is_rejected(self):
        for content in ("我觉得这个回答是对的", "[1, 2]", "", None, "```json\n{坏了}\n```"):
            with self.subTest(content=content):
                parsed, note = ev.extract_structured_data(content, Judgment)
                self.assertIsNone(parsed)
                self.assertTrue(note)


class FakeChain:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def invoke(self, messages, config=None):
        output = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        if isinstance(output, Exception):
            raise output
        return output


class FakeLLM:
    def __init__(self, outputs):
        self.chain = FakeChain(outputs)
        self.kwargs = None

    def with_structured_output(self, schema, **kwargs):
        self.kwargs = kwargs
        return self.chain


def raw(content):
    return SimpleNamespace(content=content)


def tool_output(**overrides):
    """工具调用成功时的返回形状（include_raw=True）。"""
    class Parsed:
        def model_dump(self):
            return dict(VALID, **overrides)

    return {"raw": raw(""), "parsed": Parsed(), "parsing_error": None}


class JudgeFallbackTests(unittest.TestCase):
    """三层退路：工具调用 → 正文兜底 → 重试。全部离线。"""

    def make(self, outputs):
        llm = FakeLLM(outputs)
        return ev.make_judge(SimpleNamespace(llm=llm)), llm

    def row(self):
        return {"answer": "回答", "documents": [], "citation_present": False,
                "invalid_citation_ids": []}

    def case(self, answerable=True):
        return {"question": "问", "answerable": answerable, "reference_answer": "答"}

    def test_include_raw_is_requested(self):
        # 不开这个开关，解析失败会直接抛异常，正文兜底根本没机会跑
        _, llm = self.make([tool_output()])
        self.assertEqual(llm.kwargs, {"include_raw": True})

    def test_tool_call_is_preferred(self):
        judge, _ = self.make([tool_output()])
        data = judge(self.case(), self.row(), None)
        self.assertEqual(data["judge_source"], "tool")

    def test_text_fallback_when_the_model_skips_the_tool(self):
        content = "```json\n" + json.dumps({"Judgment": VALID}, ensure_ascii=False) + "\n```"
        judge, llm = self.make([{"raw": raw(content), "parsed": None, "parsing_error": None}])
        data = judge(self.case(), self.row(), None)
        self.assertEqual(data["judge_source"], "text")
        self.assertTrue(data["answer_correct"])
        self.assertEqual(llm.chain.calls, 1)  # 兜底成功就不该再打一次接口

    def test_retry_after_a_transient_failure(self):
        judge, llm = self.make([
            RuntimeError("500 Internal Server Error"),
            {"raw": raw(json.dumps(VALID)), "parsed": None, "parsing_error": None},
        ])
        data = judge(self.case(), self.row(), None)
        self.assertEqual(data["judge_source"], "retry")
        self.assertEqual(llm.chain.calls, 2)

    def test_explicit_balance_exhaustion_is_not_retried(self):
        error = RuntimeError("429 Too Many Requests")
        error.response = SimpleNamespace(json=lambda: {"error": {"code": "1113"}})
        judge, llm = self.make([error])
        with self.assertRaises(RuntimeError):
            judge(self.case(), self.row(), None)
        self.assertEqual(llm.chain.calls, 1)

    def test_unparseable_output_raises_instead_of_faking_a_score(self):
        judge, llm = self.make([{"raw": raw("我拒绝调用工具"), "parsed": None,
                                 "parsing_error": None}])
        with self.assertRaisesRegex(ValueError, "没有返回可解析的结构化结果"):
            judge(self.case(), self.row(), None)
        self.assertEqual(llm.chain.calls, 2)  # 兜底失败 → 重试一次 → 才放弃

    def test_failure_message_keeps_the_reason_and_the_raw_output(self):
        # 排查这一轮故障绕了三个探针，就因为原来的错误信息只写"没有可解析的结果"，
        # 把失败原因和模型的原始返回全丢了。这条测试钉住这个诊断信息。
        judge, _ = self.make([{"raw": raw('{"name": "Judgment", "parameters": {}}'),
                               "parsed": None, "parsing_error": None}])
        with self.assertRaises(ValueError) as caught:
            judge(self.case(), self.row(), None)
        message = str(caught.exception)
        self.assertIn("不符合 schema", message)   # 失败原因
        self.assertIn("parameters", message)      # 原始返回的片段

    def test_failure_message_reports_the_schema_validation_error(self):
        # parsing_error 一直躺在 include_raw 的返回值里没人用。代价很具体：
        # --judge-model glm-4-plus 时所有题目只报「正文为空」，而模型其实调了工具、
        # 只是参数没过校验，正文本来就是空的——真正的原因全在 parsing_error 里。
        raw_error = (
            "2 validation errors for Judgment\n"
            "grounded\n"
            "  Input should be a valid boolean [type=bool_type, input_value='true', input_type=str]\n"
            "    For further information visit https://errors.pydantic.dev/2.13/v/bool_type\n"
            "citations_supported\n"
            "  Input should be a valid boolean [type=bool_type, input_value='true', input_type=str]"
        )
        judge, _ = self.make([{"raw": raw("\n"), "parsed": None,
                               "parsing_error": ValueError(raw_error)}])
        with self.assertRaises(ValueError) as caught:
            judge(self.case(), self.row(), None)
        message = str(caught.exception)
        self.assertIn("结构化校验失败", message)
        self.assertIn("grounded", message)
        self.assertIn("citations_supported", message)  # 第二个出错字段不能被样板文字挤掉
        self.assertNotIn("errors.pydantic.dev", message)

    def test_missing_citations_are_forced_to_null(self):
        judge, _ = self.make([tool_output(citations_supported=True)])
        data = judge(self.case(), self.row(), None)
        self.assertIsNone(data["citations_supported"])

    def test_source_is_recorded_on_the_row_and_counted_in_the_summary(self):
        def judge(case, row, tracker):
            data = dict(VALID)
            data["judge_source"] = "text"
            return data

        case = sample_case()
        row = ev.evaluate_case(target(), case, "agent", 5, judge=judge)
        self.assertEqual(row["judge_source"], "text")
        self.assertNotIn("judge_source", row["judgment"])
        result = ev.summarize([row])["agent"]
        self.assertEqual(result["judge_via_text"], 1)
        self.assertEqual(result["judge_via_retry"], 0)

    def test_inconsistent_refusal_verdict_is_flagged(self):
        # 不可回答题：裁判一边说"答对了"，一边说"没拒答"。实测旧版本 12 道题全踩这个坑，
        # 把拒答率从 100% 显示成 0%。这是自相矛盾，必须能被看见。
        def judge(case, row, tracker):
            return {"answer_correct": True, "grounded": None, "abstained": False,
                    "citations_supported": None, "reason": "…"}

        case = sample_case()
        case.update(answerable=False, reference_pages=[])
        row = ev.evaluate_case(target(), case, "agent", 5, judge=judge)
        self.assertTrue(row["judge_inconsistent"])
        self.assertIsNone(row["judge_source"])  # 假裁判没给来源
        self.assertEqual(ev.summarize([row])["agent"]["judge_inconsistent"], 1)

    def test_consistent_refusal_and_normal_answer_are_not_flagged(self):
        def refusal(case, row, tracker):
            return {"answer_correct": True, "grounded": None, "abstained": True,
                    "citations_supported": None, "reason": "明确承认无法确定"}

        def answered(case, row, tracker):
            return {"answer_correct": True, "grounded": True, "abstained": False,
                    "citations_supported": None, "reason": "给出了留出法等三种方法"}

        unanswered = sample_case()
        unanswered.update(answerable=False, reference_pages=[])
        self.assertFalse(
            ev.evaluate_case(target(), unanswered, "agent", 5, judge=refusal)["judge_inconsistent"])
        self.assertFalse(
            ev.evaluate_case(target(), sample_case(), "agent", 5, judge=answered)["judge_inconsistent"])


class JudgePromptFormatTests(unittest.TestCase):
    """裁判提示词的排版是被实测钉住的，不是审美偏好。

    同一批 6 条样本各重复 3 次，只改提示词、不改 tool_choice：
    扁平版（一行一条规则）工具调用 12~15/18、输出可解析 18/18；
    改成分段 + 缩进子项的「整齐排版」后，工具调用掉到 0/18，模型转而去复述工具定义，
    可解析只剩 7/18。所以下面这几条断言是防回归的，不要因为「看着更清楚」就放开。
    """

    def test_no_blank_lines_or_indentation(self):
        for number, line in enumerate(ev.JUDGE_SYSTEM.splitlines(), 1):
            with self.subTest(line=number):
                self.assertTrue(line.strip(), f"第 {number} 行是空行：分段会破坏工具调用")
                self.assertEqual(line, line.lstrip(),
                                 f"第 {number} 行有前导空格：缩进子项会破坏工具调用")

    def test_field_definitions_keep_the_original_order(self):
        # 用「字段名：」定位定义行：字段名在正文别处也会出现（比如那句补充说明里就有 abstained）
        names = ["answer_correct", "grounded", "abstained", "citations_supported"]
        positions = [ev.JUDGE_SYSTEM.index(f"{name}：") for name in names]
        self.assertEqual(positions, sorted(positions), "字段定义被调换过顺序")

    def test_the_two_clarifications_survive(self):
        self.assertIn("同时为 true", ev.JUDGE_SYSTEM)        # 正确拒答不是自相矛盾
        self.assertIn("candidate_answer", ev.JUDGE_SYSTEM)   # reason 要引用具体内容，不许复述规则


if __name__ == "__main__":
    unittest.main()
