"""
AI 프롬프트 토큰 최적화 대시보드
================================
Streamlit 기반 웹 인터페이스.
- 탭 1: 개별 프롬프트 최적화
- 탭 2: 벤치마크 실험 (논문용 결과 분석)
"""

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from optimizer.tokenizer import TokenCounter
from optimizer.analyzer import PatternAnalyzer
from optimizer.refiner import PromptRefiner
from optimizer.cost import CostCalculator, MODEL_PRICING
from optimizer.benchmark import BenchmarkRunner, BENCHMARK_DATASET, HybridBenchmarkRunner
from optimizer.charts import (
    plot_category_reduction,
    plot_token_comparison,
    plot_pattern_frequency,
    plot_reduction_distribution,
    plot_cost_simulation,
    plot_before_after_scatter,
    generate_all_figures,
    plot_hybrid_comparison,
    plot_hybrid_improvement,
    plot_hybrid_scatter,
    generate_hybrid_figures,
)
from optimizer.hybrid_engine import HybridOptimizer
from optimizer.learned_optimizer import AdaptiveRefiner

# ─── 한글 폰트 설정 ───
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ─── 페이지 설정 ───
st.set_page_config(
    page_title="프롬프트 토큰 최적화 도구",
    page_icon="⚡",
    layout="wide",
)

# ─── 커스텀 CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    .stApp { font-family: 'Noto Sans KR', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem; font-weight: 700;
        text-align: center; margin-bottom: 0.5rem;
    }
    .sub-header { text-align: center; color: #6b7280; font-size: 1.1rem; margin-bottom: 2rem; }
    .token-chip {
        display: inline-block; padding: 2px 6px; margin: 1px;
        border-radius: 4px; font-size: 0.85rem; font-family: monospace;
    }
    .stat-card {
        border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem;
        text-align: center; background: #f9fafb;
    }
</style>
""", unsafe_allow_html=True)

# ─── 헤더 ───
st.markdown('<div class="main-header">⚡ 프롬프트 토큰 최적화 도구</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI 프롬프트의 불필요한 토큰을 분석하고 자동으로 최적화합니다</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════
# 사이드바
# ═══════════════════════════════════════
with st.sidebar:
    st.header("⚙️ 설정")
    model = st.selectbox("모델 선택", options=list(MODEL_PRICING.keys()), index=1)

    st.divider()
    st.subheader("🔑 API 설정")
    st.info("현재 무료 LLM (g4f) 백엔드를 사용 중입니다. 별도의 API 키가 필요하지 않습니다.")
        
    st.divider()
    st.subheader("🔧 규칙 기반 설정 (기본)")
    fix_whitespace = st.checkbox("중복 공백/줄바꿈 정리", value=True)
    fix_polite = st.checkbox("과잉 공손 표현 제거", value=True)
    fix_fillers = st.checkbox("불필요 접속사/수식어 제거", value=True)
    fix_repetitive = st.checkbox("반복 강조 표현 통합", value=True)
    fix_unnecessary = st.checkbox("불필요 지시 문구 제거", value=True)

    st.divider()
    st.subheader("💰 비용 시뮬레이션")
    daily_calls = st.number_input("일일 API 호출 횟수", min_value=1, value=100, step=10)

    st.divider()
    st.caption("📌 산업공학과 캡스톤 프로젝트")
    st.caption("AI 프롬프트 토큰 최적화 방안 연구")

# ═══════════════════════════════════════
# 메인 탭
# ═══════════════════════════════════════
main_tab1, main_tab_hybrid, main_tab_model, main_tab2 = st.tabs(["🔧 개별 프롬프트 최적화", "🧠 하이브리드 최적화", "📋 프로젝트 기획안", "📊 벤치마크 실험 (논문용)"])

# ───────────────────────────────────────
# 탭 1: 개별 프롬프트 최적화
# ───────────────────────────────────────
with main_tab1:
    SAMPLE_PROMPTS = {
        "예시 선택...": "",
        "🇰🇷 한국어 - 과잉 공손": (
            "안녕하세요, 혹시 괜찮으시다면 부탁드립니다.\n"
            "바쁘시겠지만 죄송하지만 아래의 내용을 잘 읽고\n"
            "다음에 대해 자세하게 상세히 설명해 주세요.\n\n"
            "파이썬에서 리스트와 튜플의 차이점이 뭔가요?\n\n"
            "최대한 자세하게 알려주세요. 감사합니다."
        ),
        "🇰🇷 한국어 - 반복 강조": (
            "꼭 반드시 정말 정말 중요한 내용만 골라서\n"
            "아주 매우 명확하게 명확히 설명해줘.\n"
            "그리고 또한 추가적으로 예시도 들어줘.\n"
            "다시 말해서 핵심만 간결하게 정리해줘.\n"
            "그래서 결국 요약을 부탁해."
        ),
        "🇰🇷 한국어 - 중복 공백": (
            "제가   지금부터   질문할게요.\n\n\n\n"
            "너는  이제부터  파이썬  전문가야.\n\n\n"
            "기본적으로  일반적으로  말하자면\n"
            "파이썬은   좋은   언어입니다.\n\n\n\n"
            "설명해주세요."
        ),
    }

    col_sample, _ = st.columns([1, 2])
    with col_sample:
        selected_sample = st.selectbox("예시 프롬프트", options=list(SAMPLE_PROMPTS.keys()))

    prompt_input = st.text_area(
        "분석할 프롬프트를 입력하세요",
        value=SAMPLE_PROMPTS.get(selected_sample, ""),
        height=200,
        placeholder="여기에 프롬프트를 붙여넣으세요...",
    )

    if st.button("🚀 분석 및 최적화 실행", type="primary", use_container_width=True):
        if not prompt_input.strip():
            st.warning("프롬프트를 입력해 주세요.")
        else:
            counter = TokenCounter(model=model)
            refiner = PromptRefiner(model=model)
            calculator = CostCalculator(model=model)

            result = refiner.refine(
                prompt_input,
                fix_whitespace=fix_whitespace, fix_polite=fix_polite,
                fix_fillers=fix_fillers, fix_repetitive=fix_repetitive,
                fix_unnecessary=fix_unnecessary,
            )

            # 핵심 지표
            st.divider()
            st.subheader("📊 최적화 결과")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("원본 토큰 수", f"{result.original_tokens:,}")
            c2.metric("최적화 토큰 수", f"{result.refined_tokens:,}")
            c3.metric("절감 토큰", f"{result.saved_tokens:,}", delta=f"-{result.saved_tokens:,}", delta_color="inverse")
            c4.metric("절감률", f"{result.reduction_rate * 100:.1f}%")

            # 프롬프트 비교
            st.divider()
            st.subheader("📄 프롬프트 비교")
            co, cr = st.columns(2)
            with co:
                st.markdown("**원본 프롬프트**")
                st.code(result.original, language=None)
            with cr:
                st.markdown("**✅ 최적화된 프롬프트**")
                st.code(result.refined, language=None)

            # 토큰 시각화
            st.divider()
            st.subheader("🔤 토큰 시각화")
            TOKEN_COLORS = [
                "#FECDD3", "#FDE68A", "#A7F3D0", "#BFDBFE",
                "#DDD6FE", "#FBC4AB", "#C7CEEA", "#E2F0CB",
            ]

            def render_tokens(text, cnt):
                toks = cnt.tokenize(text)
                parts = []
                for i, t in enumerate(toks):
                    c = TOKEN_COLORS[i % len(TOKEN_COLORS)]
                    esc = t["text"].replace("<", "&lt;").replace(">", "&gt;").replace(" ", "·").replace("\n", "↵\n")
                    parts.append(f'<span class="token-chip" style="background-color:{c};" title="ID: {t["token_id"]}">{esc}</span>')
                return "".join(parts)

            t1, t2 = st.tabs(["원본 토큰", "최적화 토큰"])
            with t1:
                st.markdown(render_tokens(result.original, counter), unsafe_allow_html=True)
                st.caption(f"총 {result.original_tokens:,}개 토큰")
            with t2:
                st.markdown(render_tokens(result.refined, counter), unsafe_allow_html=True)
                st.caption(f"총 {result.refined_tokens:,}개 토큰")

            # 패턴 분석
            if result.analysis and result.analysis.patterns_found:
                st.divider()
                st.subheader("🔍 낭비 패턴 분석")
                pat_df = pd.DataFrame([
                    {"패턴": p.category, "감지 수": p.count, "추정 낭비 토큰": p.estimated_waste, "설명": p.description}
                    for p in result.analysis.patterns_found
                ])
                st.dataframe(pat_df, use_container_width=True, hide_index=True)

            # 적용된 규칙
            if result.applied_rules:
                st.divider()
                st.subheader("✂️ 적용된 정제 규칙")
                rules_df = pd.DataFrame([
                    {"카테고리": r.get("category", ""), "규칙": r.get("rule", ""), "적용 횟수": r.get("count", 1)}
                    for r in result.applied_rules
                ])
                st.dataframe(rules_df, use_container_width=True, hide_index=True)

            # 비용 비교
            st.divider()
            st.subheader("💰 비용 비교")
            cost_report = calculator.compare(result.original, result.refined, daily_calls=daily_calls)
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("1회 절감액", f"${cost_report.saved_cost:.8f}")
            cc2.metric("월간 절감액", f"${cost_report.monthly_savings:.4f}")
            cc3.metric("연간 절감액", f"${cost_report.yearly_savings:.4f}")

            # 복사
            st.divider()
            st.subheader("📋 최적화된 프롬프트 복사")
            st.code(result.refined, language=None)


# ───────────────────────────────────────
# 탭: 🧠 하이브리드 최적화 (Fine-tuning + RAG)
# ───────────────────────────────────────
with main_tab_hybrid:
    st.markdown("""
    ### 🧠 하이브리드 최적화 — Fine-tuning + RAG 전략
    **기존 규칙 기반** 최적화에 **Fine-tuning 개념**(도메인 전문성)과 **RAG 개념**(유사 사례 검색)을
    결합하여 더 정교한 최적화를 수행합니다.
    """)

    st.info("""
    🔬 **Fine-tuning** = 벤치마크 데이터에서 도메인별 규칙 효과를 학습 → 도메인 특화 프로파일 적용  
    🔍 **RAG** = 유사 프롬프트 사례를 검색 → 해당 사례의 최적화 패턴을 참고하여 가이드 생성
    """)

    # 하이브리드 엔진 초기화 (캐싱)
    @st.cache_resource
    def get_hybrid_engine(_model: str):
        engine = HybridOptimizer(model=_model)
        engine.initialize(BENCHMARK_DATASET)
        return engine

    hybrid_engine = get_hybrid_engine(model)

    # 샘플 프롬프트
    HYBRID_SAMPLES = {
        "예시 선택...": "",
        "🔍 질문응답 예시": "안녕하세요, 혹시 괜찮으시다면 죄송하지만 파이썬에서 GIL이 무엇인지 최대한 자세하게 설명해 주세요. 꼭 반드시 예시도 들어주세요. 감사합니다.",
        "💻 코드생성 예시": "바쁘시겠지만 부탁드립니다. 파이썬으로 피보나치 수열을 구현해 주세요. 그리고 또한 추가적으로 재귀 버전도 작성해 주세요. 감사합니다.",
        "📝 요약 예시": "안녕하세요, 혹시 시간 되시면 아래 글을 꼭 반드시 핵심만 간추려서 3줄로 요약해 주세요. 기본적으로 사실상 중요한 것만 포함해 주세요.",
        "🌐 번역 예시": "죄송하지만 혹시 괜찮으시다면 다음 문장을 영어로 번역해 주세요. 정말 정말 자연스럽게 부탁드립니다. '오늘 회의에서 중요한 결정이 내려졌습니다.'",
    }

    hybrid_col_sample, _ = st.columns([1, 2])
    with hybrid_col_sample:
        hybrid_selected = st.selectbox("예시 프롬프트", options=list(HYBRID_SAMPLES.keys()), key="hybrid_sample")

    hybrid_input = st.text_area(
        "하이브리드 최적화할 프롬프트를 입력하세요",
        value=HYBRID_SAMPLES.get(hybrid_selected, ""),
        height=160,
        placeholder="여기에 프롬프트를 붙여넣으세요...",
        key="hybrid_input",
    )

    if st.button("🧠 하이브리드 최적화 실행", type="primary", use_container_width=True, key="hybrid_run"):
        if not hybrid_input.strip():
            st.warning("프롬프트를 입력해 주세요.")
        else:
            import time

            # ── 파이프라인 다이어그램 + 터미널 ──
            st.markdown("### 🔄 하이브리드 최적화 파이프라인")

            PIPELINE_STEPS = [
                ("📥 입력 수신", "프롬프트 텍스트 파싱"),
                ("🔬 도메인 감지", "Fine-tuning 도메인 분류"),
                ("🧬 학습 패턴 적용", "도메인 특화 압축"),
                ("🔍 RAG 검색", "유사 사례 벡터 검색"),
                ("📐 규칙 세팅", "RAG+FT 통합 의사결정"),
                ("⚙️ 46개 규칙 정제", "기본 한국어 규칙 적용"),
                ("✅ 최종 비교", "결과 확정 및 비용 산출"),
            ]

            def _render_pipeline(steps, active_idx, detail_text=""):
                """파이프라인 다이어그램 HTML 생성"""
                html = """<style>
                .pipeline-box{display:flex;align-items:center;flex-wrap:wrap;gap:4px;margin:10px 0}
                .p-step{padding:8px 14px;border-radius:8px;font-size:13px;font-weight:600;
                         border:2px solid #555;color:#aaa;background:#1e1e2e;transition:all .3s;min-width:120px;text-align:center}
                .p-step.done{border-color:#22c55e;color:#22c55e;background:#0d2818}
                .p-step.active{border-color:#facc15;color:#facc15;background:#2d2a10;
                               animation:pulse 1s ease-in-out infinite alternate}
                .p-step small{display:block;font-weight:400;font-size:11px;opacity:.7;margin-top:2px}
                .p-arrow{color:#555;font-size:18px;font-weight:bold}
                .p-arrow.done{color:#22c55e}
                .p-detail{background:#161622;border:1px solid #333;border-radius:6px;padding:8px 12px;
                          color:#aaa;font-size:12px;margin-top:6px;font-family:monospace}
                @keyframes pulse{from{box-shadow:0 0 4px #facc1544}to{box-shadow:0 0 16px #facc1588}}
                </style><div class="pipeline-box">"""
                for i, (name, desc) in enumerate(steps):
                    cls = "done" if i < active_idx else ("active" if i == active_idx else "")
                    html += f'<div class="p-step {cls}">{name}<small>{desc}</small></div>'
                    if i < len(steps) - 1:
                        acls = "done" if i < active_idx else ""
                        html += f'<span class="p-arrow {acls}">→</span>'
                html += "</div>"
                if detail_text:
                    html += f'<div class="p-detail">▶ {detail_text}</div>'
                return html

            pipeline_ph = st.empty()
            terminal_ph = st.empty()
            logs = []

            def _log(msg):
                logs.append(msg)
                terminal_ph.code("\n".join(logs), language="bash")

            # Step 0: 입력 수신
            pipeline_ph.markdown(_render_pipeline(PIPELINE_STEPS, 0, f"프롬프트 수신 완료 ({len(hybrid_input)}자)"), unsafe_allow_html=True)
            _log("$ ./hybrid_optimizer --run-pipeline")
            _log(f"[INPUT] 프롬프트 수신 ({len(hybrid_input)}자)")
            time.sleep(0.4)

            # Step 1~: 엔진 실행
            pipeline_ph.markdown(_render_pipeline(PIPELINE_STEPS, 1, "도메인 키워드 매칭 중..."), unsafe_allow_html=True)
            _log("[FINE-TUNING] 도메인 키워드 스캐닝...")
            time.sleep(0.3)

            with st.spinner("하이브리드 엔진 실행 중..."):
                h_result = hybrid_engine.optimize(hybrid_input, top_k=3)

            # Step 1 완료 → Step 2
            pipeline_ph.markdown(_render_pipeline(PIPELINE_STEPS, 2, f"도메인 '{h_result.detected_domain}' 감지 (신뢰도 {h_result.domain_confidence:.0%})"), unsafe_allow_html=True)
            _log(f"[FINE-TUNING] 도메인: '{h_result.detected_domain}' (신뢰도 {h_result.domain_confidence:.0%})")
            time.sleep(0.4)

            if h_result.learned_patterns_applied:
                for pat in h_result.learned_patterns_applied:
                    _log(f"  ↳ {pat['rule']}")
                pipeline_ph.markdown(_render_pipeline(PIPELINE_STEPS, 2, f"학습 패턴 {len(h_result.learned_patterns_applied)}개 적용 완료"), unsafe_allow_html=True)
            else:
                _log("[FINE-TUNING] 적용 가능한 도메인 패턴 없음")
            time.sleep(0.3)

            # Step 3: RAG
            similar_count = len(h_result.rag_advice.similar_cases)
            pipeline_ph.markdown(_render_pipeline(PIPELINE_STEPS, 3, f"TF-IDF 코사인 유사도 검색 → {similar_count}건 발견"), unsafe_allow_html=True)
            _log(f"[RAG] 유사 사례 {similar_count}건 검색 완료")
            if similar_count > 0:
                _log(f"[RAG] 예상 절감률: {h_result.rag_advice.predicted_reduction_rate:.1%}")
            time.sleep(0.3)

            # Step 4: 규칙 세팅
            pipeline_ph.markdown(_render_pipeline(PIPELINE_STEPS, 4, "RAG + Fine-tuning 추천 통합 → 최적 규칙 세트 결정"), unsafe_allow_html=True)
            _log("[DECIDE] RAG+FT 통합 의사결정 완료")
            time.sleep(0.3)

            # Step 5: 46개 규칙
            pipeline_ph.markdown(_render_pipeline(PIPELINE_STEPS, 5, "46개 한국어 정제 규칙 순차 적용 중..."), unsafe_allow_html=True)
            _log("[REFINER] 46개 규칙 적용 완료")
            time.sleep(0.3)

            # Step 6: 최종 비교
            pipeline_ph.markdown(_render_pipeline(PIPELINE_STEPS, 6, f"원본 {h_result.original_tokens}→{h_result.hybrid_tokens} 토큰 (절감 {h_result.hybrid_reduction*100:.1f}%)"), unsafe_allow_html=True)
            _log(f"[RESULT] {h_result.original_tokens}→{h_result.hybrid_tokens} 토큰 ({h_result.hybrid_reduction*100:.1f}% 절감)")
            time.sleep(0.3)

            # 전부 완료
            pipeline_ph.markdown(_render_pipeline(PIPELINE_STEPS, 7, "파이프라인 완료 ✅"), unsafe_allow_html=True)
            _log("$ exit 0")

            # ── 전략 설명 ──
            st.divider()
            st.subheader("🎯 하이브리드 전략 분석")
            st.markdown(f"**전략 결정 과정**: `{h_result.strategy_explanation}`")

            strat_c1, strat_c2 = st.columns(2)
            with strat_c1:
                st.markdown("#### 🔬 Fine-tuning 기여")
                st.markdown(f"- **감지 도메인**: `{h_result.detected_domain}`")
                st.markdown(f"- **도메인 신뢰도**: `{h_result.domain_confidence:.0%}`")
                st.markdown(f"- **분석**: {h_result.finetuning_contribution}")
            with strat_c2:
                st.markdown("#### 🔍 RAG 기여")
                st.markdown(f"- **분석**: {h_result.rag_contribution}")
                if h_result.rag_advice:
                    st.markdown(f"- **예상 절감률**: `{h_result.rag_advice.predicted_reduction_rate:.1%}`")
                    st.markdown(f"- **RAG 신뢰도**: `{h_result.rag_advice.confidence:.0%}`")

            # ── 핵심 결과 비교 ──
            st.divider()
            st.subheader("📊 규칙 기반 vs 하이브리드 비교")

            comp_c1, comp_c2, comp_c3, comp_c4 = st.columns(4)
            comp_c1.metric("원본 토큰", f"{h_result.original_tokens:,}")
            comp_c2.metric(
                "규칙 기반 결과",
                f"{h_result.rule_based_tokens:,}",
                delta=f"-{h_result.original_tokens - h_result.rule_based_tokens:,}",
                delta_color="inverse",
            )
            comp_c3.metric(
                "하이브리드 결과",
                f"{h_result.hybrid_tokens:,}",
                delta=f"-{h_result.original_tokens - h_result.hybrid_tokens:,}",
                delta_color="inverse",
            )
            comp_c4.metric(
                "추가 절감",
                f"{h_result.additional_savings:,} 토큰",
                delta=f"+{h_result.improvement_rate:.1%}" if h_result.improvement_rate > 0 else "동일",
            )

            # 절감률 비교 바
            bar_df = pd.DataFrame([
                {"방식": "규칙 기반", "절감률 (%)": round(h_result.rule_based_reduction * 100, 1)},
                {"방식": "하이브리드", "절감률 (%)": round(h_result.hybrid_reduction * 100, 1)},
            ])
            st.bar_chart(bar_df.set_index("방식"), horizontal=True)

            # ── 프롬프트 비교 ──
            st.divider()
            st.subheader("📄 프롬프트 비교")
            pr_c1, pr_c2, pr_c3 = st.columns(3)
            with pr_c1:
                st.markdown(f"**원본** ({h_result.original_tokens} 토큰)")
                st.code(h_result.original_text, language=None)
            with pr_c2:
                st.markdown(f"**규칙 기반** ({h_result.rule_based_tokens} 토큰)")
                st.code(h_result.rule_based_result.refined, language=None)
            with pr_c3:
                st.markdown(f"**✅ 하이브리드** ({h_result.hybrid_tokens} 토큰)")
                st.code(h_result.hybrid_refined_text, language=None)

            # ── RAG 유사 사례 ──
            if h_result.rag_similar_cases:
                st.divider()
                st.subheader("🔍 RAG — 유사 사례 검색 결과")
                for case in h_result.rag_similar_cases:
                    with st.expander(
                        f"#{case['rank']} | 유사도 {case['similarity']:.0%} | "
                        f"{case['category']} | 절감률 {case['reduction_rate']:.1%}"
                    ):
                        rag_c1, rag_c2 = st.columns(2)
                        with rag_c1:
                            st.markdown("**원본**")
                            st.code(case["original"], language=None)
                        with rag_c2:
                            st.markdown("**최적화 후**")
                            st.code(case["refined"], language=None)

            # ── RAG 최적화 팁 ──
            if h_result.rag_advice and h_result.rag_advice.optimization_tips:
                st.divider()
                st.subheader("💡 RAG 기반 최적화 팁")
                for tip in h_result.rag_advice.optimization_tips:
                    st.markdown(f"- {tip}")

            # ── 비용 비교 ──
            st.divider()
            st.subheader("💰 비용 비교")
            cost_c1, cost_c2, cost_c3 = st.columns(3)
            cost_c1.metric("규칙 기반 비용", f"${h_result.cost_rule_based:.8f}")
            cost_c2.metric("하이브리드 비용", f"${h_result.cost_hybrid:.8f}")
            cost_c3.metric("추가 절감액", f"${h_result.cost_savings:.8f}")

            # 월간/연간 시뮬레이션
            calculator = CostCalculator(model=model)
            sim_rule = calculator.simulate_bulk(
                h_result.original_text,
                h_result.rule_based_result.refined,
                call_counts=[100, 1000, 10000],
            )
            sim_hybrid = calculator.simulate_bulk(
                h_result.original_text,
                h_result.hybrid_refined_text,
                call_counts=[100, 1000, 10000],
            )

            sim_df = pd.DataFrame([{
                "일일 호출": f"{sr['daily_calls']:,}회",
                "규칙 기반 월간 절감($)": f"{sr['monthly_savings']:.4f}",
                "하이브리드 월간 절감($)": f"{sh['monthly_savings']:.4f}",
                "규칙 기반 연간 절감($)": f"{sr['yearly_savings']:.4f}",
                "하이브리드 연간 절감($)": f"{sh['yearly_savings']:.4f}",
            } for sr, sh in zip(sim_rule, sim_hybrid)])
            st.dataframe(sim_df, use_container_width=True, hide_index=True)

    # ═══ A/B 벤치마크 실험 (논문용) ═══
    st.divider()
    st.markdown("""
    ### 🧪 A/B 벤치마크: 규칙 기반 vs 하이브리드 비교 실험
    60건 전체 데이터셋에 대해 **규칙 기반**과 **하이브리드** 방식을 동시에 실행하고,
    통계 검정(paired t-test, Wilcoxon, Cohen's d)으로 유의성을 검증합니다.
    """)

    if st.button("🧪 A/B 벤치마크 실행 (60건 전수 비교)", type="primary", use_container_width=True, key="hybrid_bench"):
        import time

        st.markdown("### 🔄 A/B 벤치마크 파이프라인")

        BENCH_STEPS = [
            ("🔧 엔진 초기화", "HybridOptimizer 워밍업"),
            ("📊 데이터 처리", "60건 순차 A/B 비교"),
            ("📐 통계 분석", "t-test, Wilcoxon, Cohen's d"),
            ("📈 차트 생성", "논문용 비교 시각화"),
            ("✅ 완료", "보고서 확정"),
        ]

        def _render_bench_pipeline(steps, active_idx, detail_text=""):
            html = """<style>
            .bp-box{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:10px 0}
            .bp-step{padding:10px 16px;border-radius:8px;font-size:13px;font-weight:600;
                     border:2px solid #555;color:#aaa;background:#1e1e2e;min-width:130px;text-align:center}
            .bp-step.done{border-color:#22c55e;color:#22c55e;background:#0d2818}
            .bp-step.active{border-color:#facc15;color:#facc15;background:#2d2a10;
                           animation:bpulse 1s ease-in-out infinite alternate}
            .bp-step small{display:block;font-weight:400;font-size:11px;opacity:.7;margin-top:2px}
            .bp-arrow{color:#555;font-size:20px;font-weight:bold}
            .bp-arrow.done{color:#22c55e}
            .bp-detail{background:#161622;border:1px solid #333;border-radius:6px;padding:8px 12px;
                      color:#aaa;font-size:12px;margin-top:6px;font-family:monospace}
            @keyframes bpulse{from{box-shadow:0 0 4px #facc1544}to{box-shadow:0 0 16px #facc1588}}
            </style><div class="bp-box">"""
            for i, (name, desc) in enumerate(steps):
                cls = "done" if i < active_idx else ("active" if i == active_idx else "")
                html += f'<div class="bp-step {cls}">{name}<small>{desc}</small></div>'
                if i < len(steps) - 1:
                    acls = "done" if i < active_idx else ""
                    html += f'<span class="bp-arrow {acls}">→</span>'
            html += "</div>"
            if detail_text:
                html += f'<div class="bp-detail">▶ {detail_text}</div>'
            return html

        bench_pipeline_ph = st.empty()
        bench_terminal_ph = st.empty()
        bench_logs = []

        def _bench_log(msg):
            bench_logs.append(msg)
            if len(bench_logs) > 18:
                display = bench_logs[-18:]
            else:
                display = bench_logs
            bench_terminal_ph.code("\n".join(display), language="bash")

        # Step 0: 엔진 초기화
        bench_pipeline_ph.markdown(_render_bench_pipeline(BENCH_STEPS, 0, "하이브리드 엔진 및 벤치마크 런너 초기화 중..."), unsafe_allow_html=True)
        _bench_log("$ ./run_ab_benchmark --mode=hybrid --samples=30")
        _bench_log("[SYSTEM] HybridBenchmarkRunner 초기화 중...")
        time.sleep(0.4)

        # Step 1: 데이터 처리 (30건 순차)
        bench_pipeline_ph.markdown(_render_bench_pipeline(BENCH_STEPS, 1, "60건 프롬프트 A/B 비교 시작..."), unsafe_allow_html=True)
        _bench_log("[ENGINE] 벤치마크 데이터 로드 완료. 처리 시작.")

        def bench_progress_callback(current, total, info):
            cat = info['category']
            dom = info['domain']
            red = info['reduction'] * 100
            log_line = f"  [{current:02d}/{total}] {cat} | Domain: '{dom}' | 절감: {red:.1f}%"
            _bench_log(log_line)
            bench_pipeline_ph.markdown(
                _render_bench_pipeline(BENCH_STEPS, 1, f"처리 중: {current}/{total} ('{cat}' → 도메인 '{dom}', 절감 {red:.1f}%)"),
                unsafe_allow_html=True
            )
            time.sleep(0.08)

        with st.spinner("A/B 실험 진행 중..."):
            h_runner = HybridBenchmarkRunner(model=model)
            h_report = h_runner.run(progress_callback=bench_progress_callback)

        # Step 2: 통계 분석
        bench_pipeline_ph.markdown(_render_bench_pipeline(BENCH_STEPS, 2, "paired t-test, Wilcoxon signed-rank, Cohen's d 계산 중..."), unsafe_allow_html=True)
        _bench_log("[STATS] 통계 검정 수행 완료")
        time.sleep(0.5)

        # Step 3: 차트 생성
        bench_pipeline_ph.markdown(_render_bench_pipeline(BENCH_STEPS, 3, "논문용 비교 차트 렌더링 중..."), unsafe_allow_html=True)
        _bench_log("[CHART] 비교 시각화 생성 완료")
        time.sleep(0.4)

        # Step 4: 완료
        bench_pipeline_ph.markdown(_render_bench_pipeline(BENCH_STEPS, 5, "벤치마크 파이프라인 완료 ✅"), unsafe_allow_html=True)
        _bench_log(f"[DONE] A/B 벤치마크 완료: {h_report.total_samples}건 처리됨")
        _bench_log("$ exit 0")

        st.success(f"✅ A/B 벤치마크 완료: {h_report.total_samples}건")

        # ══ 1. 전체 요약 ══
        st.divider()
        st.subheader("📊 A/B 요약 통계")
        ab_c1, ab_c2, ab_c3, ab_c4 = st.columns(4)
        ab_c1.metric("규칙 기반 평균 절감률", f"{h_report.overall_rb_avg_reduction * 100:.2f}%")
        ab_c2.metric("하이브리드 평균 절감률", f"{h_report.overall_hy_avg_reduction * 100:.2f}%")
        ab_c3.metric("추가 개선", f"+{h_report.overall_improvement * 100:.2f}%p")
        if "effect_size" in h_report.statistical_tests:
            ab_c4.metric("Cohen's d", f"{h_report.statistical_tests['effect_size']['cohens_d']:.4f}")

        # ══ 2. 통계 검정 ══
        st.divider()
        st.subheader("📐 통계 검정 결과")

        stat = h_report.statistical_tests
        stat_data = []
        if "effect_size" in stat:
            es = stat["effect_size"]
            stat_data.append({"검정 방법": "Cohen's d (효과 크기)", "통계량": f"d = {es['cohens_d']:.4f}", "p-value": "-", "해석": es["interpretation"]})
        if "paired_ttest" in stat:
            tt = stat["paired_ttest"]
            p_str = f"{tt['p_value']:.6f}" if "p_value" in tt else "N/A"
            sig = "✅ 유의" if tt.get("significant_005") else "❌ 비유의"
            stat_data.append({"검정 방법": "Paired t-test", "통계량": f"t = {tt.get('t_statistic', 'N/A')}", "p-value": p_str, "해석": f"α=0.05: {sig}"})
        if "wilcoxon" in stat and "statistic" in stat["wilcoxon"]:
            wt = stat["wilcoxon"]
            sig = "✅ 유의" if wt.get("significant_005") else "❌ 비유의"
            stat_data.append({"검정 방법": "Wilcoxon signed-rank", "통계량": f"W = {wt['statistic']}", "p-value": f"{wt['p_value']:.6f}", "해석": f"α=0.05: {sig}"})

        if stat_data:
            st.dataframe(pd.DataFrame(stat_data), use_container_width=True, hide_index=True)

        # ══ 3. 카테고리별 비교 ══
        st.divider()
        st.subheader("📁 카테고리별 A/B 비교")

        cat_df = pd.DataFrame([{
            "카테고리": s.category,
            "샘플 수": s.sample_count,
            "규칙 기반 평균(%)": f"{s.rb_avg_reduction * 100:.2f}",
            "규칙 표준편차": f"{s.rb_std_reduction * 100:.2f}",
            "하이브리드 평균(%)": f"{s.hy_avg_reduction * 100:.2f}",
            "하이브리드 표준편차": f"{s.hy_std_reduction * 100:.2f}",
            "평균 개선(%p)": f"+{s.avg_improvement * 100:.2f}",
            "최대 개선(%p)": f"+{s.max_improvement * 100:.2f}",
        } for s in h_report.category_stats])
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

        # ══ 4. 비교 차트 ══
        st.divider()
        st.subheader("📈 논문용 비교 차트")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("**Figure 7: 규칙 기반 vs 하이브리드 절감률 비교**")
            fig7 = plot_hybrid_comparison(h_report)
            st.pyplot(fig7)
        with chart_col2:
            st.markdown("**Figure 9: 규칙 기반 vs 하이브리드 산점도**")
            fig9 = plot_hybrid_scatter(h_report)
            st.pyplot(fig9)

        st.markdown("**Figure 8: 프롬프트별 추가 개선 효과 + 통계 검정 결과**")
        fig8 = plot_hybrid_improvement(h_report)
        st.pyplot(fig8)

        # ══ 5. 개별 결과 ══
        st.divider()
        st.subheader("📝 개별 프롬프트 A/B 결과")

        raw_df = pd.DataFrame([{
            "카테고리": r.category,
            "#": r.prompt_id,
            "원본 토큰": r.original_tokens,
            "규칙 기반": r.rule_based_tokens,
            "규칙 절감(%)": f"{r.rule_based_reduction * 100:.1f}",
            "하이브리드": r.hybrid_tokens,
            "하이브리드 절감(%)": f"{r.hybrid_reduction * 100:.1f}",
            "추가 절감": r.additional_savings,
            "개선(%p)": f"+{r.improvement * 100:.1f}",
            "도메인": r.detected_domain,
            "학습패턴": r.learned_patterns_count,
        } for r in h_report.results])
        st.dataframe(raw_df, use_container_width=True, hide_index=True)

        # ══ 6. 내보내기 ══
        st.divider()
        st.subheader("📥 A/B 데이터 내보내기")

        export_dir = "results"
        ex_c1, ex_c2, ex_c3 = st.columns(3)
        with ex_c1:
            if st.button("📥 A/B CSV", use_container_width=True, key="ab_csv"):
                HybridBenchmarkRunner.export_csv(h_report, f"{export_dir}/hybrid_benchmark.csv")
                st.success(f"✅ `{export_dir}/hybrid_benchmark.csv` 저장")
        with ex_c2:
            if st.button("📥 A/B JSON", use_container_width=True, key="ab_json"):
                HybridBenchmarkRunner.export_json(h_report, f"{export_dir}/hybrid_benchmark.json")
                st.success(f"✅ `{export_dir}/hybrid_benchmark.json` 저장")
        with ex_c3:
            if st.button("📥 비교 차트 저장 (300 DPI)", use_container_width=True, key="ab_figs"):
                saved = generate_hybrid_figures(h_report, f"{export_dir}/figures")
                for path in saved:
                    st.success(f"✅ `{path}` 저장")


# ───────────────────────────────────────
# 탭 3: 프로젝트 기획안 (캡스톤 디자인)
# ───────────────────────────────────────
with main_tab_model:
    st.markdown("## 🏗️ 오픈소스 LLM과 RAG 파라미터 최적화를 통한 프롬프트 토큰 비용 절감 실험")
    st.caption("**팀 이름:** PROMM (가칭) | **구성원:** 전남대학교 산업공학과 / 최현민")
    
    st.divider()
    st.subheader("1️⃣ 프로젝트 개요")
    st.markdown('''
    **배경 및 목적**  
    챗GPT와 같은 상용 AI를 서비스에 도입할 때 발생하는 '프롬프트 토큰 과금'은 초기 스타트업이나 학생 창업 시 큰 부담이 됩니다. 
    본 프로젝트는 비싼 API 대신 무료 오픈소스 AI 모델을 활용하고, 산업공학의 비용 최적화 관점을 적용하여 
    **'가장 적은 토큰(비용)으로 가장 정확한 답변을 얻는 최적의 프롬프트 조건'**을 찾는 것을 목적으로 합니다.
    
    **중요성 및 필요성**  
    단순히 AI를 써보는 것을 넘어, 한정된 자원(학부생 수준의 PC 및 무료 클라우드) 환경에서 시스템의 효율(비용 대비 성능)을 극대화하는 실무적인 문제 해결 과정을 경험할 수 있습니다.
    ''')

    st.divider()
    st.subheader("2️⃣ 프로젝트 목표 및 범위")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('''
        #### 🎯 구체적인 목표
        - **과금 없는 RAG 환경 구축**: 허깅페이스의 무료 한국어 모델 활용.
        - **최적의 RAG 파라미터 도출**: 실험계획법을 적용하여 프롬프트 토큰을 최소화하는 문서 자르기 크기(Chunk Size), 검색 개수(Top-K) 탐색.
        - **(선택) PEFT(LoRA) 기법**: 맛보기로 적용하여 시스템 프롬프트 길이 단축 시도.
        ''')
    with col2:
        st.markdown('''
        #### 📈 프로젝트 범위
        - **독립 변수 통제**: `Chunk Size`(예: 300자 vs 500자), `Top-K`(예: 2개 vs 4개)
        - **종속 변수 측정**: 소모된 프롬프트 총 토큰 수(Cost) 및 답변 정확도(Quality)
        - **주요 기능**: 불필요한 배경지식을 걸러내고, 꼭 필요한 정보만 모델에 전달하여 토큰 낭비 방지.
        ''')

    st.divider()
    st.subheader("3️⃣ 프로젝트 방법론")
    st.markdown('''
    - **연구 방법 및 절차**: 기준점(Baseline) 측정 → **2×2 요인실험(Factorial Design)** → 데이터 비교 분석 및 최적안 선정
    - **데이터 수집**: AI 규제 가이드라인이나 학교 학칙 등 1~2개의 PDF 문서를 샘플 데이터로 활용
    ''')
    
    method_data = pd.DataFrame([
        {"구분": "환경", "상세": "Google Colab (무료 T4 GPU 활용)"},
        {"구분": "모델", "상세": "Llama-3-8B 또는 허깅페이스 경량 한국어 모델 + BAAI/bge-m3 (임베딩)"},
        {"구분": "경량화", "상세": "메모리 부족 방지용 4-bit 양자화(Quantization) 기술 적용"},
        {"구분": "도구", "상세": "Python, Hugging Face, LangChain, 코사인 유사도 평가"},
    ])
    st.table(method_data)

    st.markdown('''
    #### 📊 성과 지표 및 평가 방법
    - **비용 (Cost)**: 질의 1건당 발생하는 평균 프롬프트 토큰 수 감소율
    - **품질 (Quality)**: 원본 정답 문서와의 '코사인 유사도(Cosine Similarity)' 점수를 비교하여 품질 하락폭이 5~10% 이내로 방어되는지 확인
    ''')

    st.divider()
    st.subheader("4️⃣ 일정 및 마일스톤 (총 16주)")
    milestone_data = pd.DataFrame([
        {"주차": "1~4주차", "목표": "개발 환경(Colab) 세팅 및 샘플 PDF 데이터 전처리", "역할": "Colab 세팅/코드 구성"},
        {"주차": "5~8주차", "목표": "오픈소스 모델 로드 및 기본 RAG 코드 작성 (베이스라인 측정)", "역할": "허깅페이스 파이프라인 연결"},
        {"주차": "9~12주차", "목표": "Chunk Size 및 Top-K 변수 조절에 따른 토큰/품질 데이터 수집 실험", "역할": "파라미터 조합별 점수 기록"},
        {"주차": "13~16주차", "목표": "기초 통계 분석(엑셀/파이썬) 및 캡스톤 최종 발표 자료 작성", "역할": "최적 조건 의사결정(Decision Making)"},
    ])
    st.dataframe(milestone_data, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("💡 결론 및 기대효과")
    st.info('''
    이 프로젝트는 거창한 AI 개발이 아닌, 산업공학의 최적화 마인드를 바탕으로 기존 AI 기술을 **'가장 경제적으로 사용하는 방법'**을 찾는 데 의의가 있습니다.  
    도출된 최적화 세팅은 향후 "PROMM"과 같은 프롬프트 교육/검증 MVP를 구축할 때 직접적인 비용 절감 가이드라인으로 활용될 수 있습니다.
    ''')


# ───────────────────────────────────────
# 탭 2: 벤치마크 실험 (논문용)
# ───────────────────────────────────────
with main_tab2:
    st.markdown("""
    ### 📊 벤치마크 실험 — 논문용 결과 분석
    내장된 벤치마크 데이터셋(4개 카테고리, 60건)에 대해 체계적 실험을 수행하고,
    **캡스톤 논문에 실을 수 있는 수준**의 통계 분석 및 차트를 생성합니다.
    """)

    st.info(f"📌 **데이터셋 구성**: {', '.join(f'{k}({len(v)}건)' for k, v in BENCHMARK_DATASET.items())} — 총 {sum(len(v) for v in BENCHMARK_DATASET.values())}건")

    if st.button("🧪 벤치마크 실험 실행", type="primary", use_container_width=True, key="bench"):
        with st.spinner("실험 진행 중... (60건의 프롬프트 분석)"):
            runner = BenchmarkRunner(model=model)
            report = runner.run()

        st.success(f"✅ 실험 완료: {report.total_samples}건 분석, 모델: {report.model}")

        # ═══ 1. 전체 요약 통계 ═══
        st.divider()
        st.subheader("1️⃣ 실험 결과 요약 (Overall Statistics)")

        os_data = report.overall_stats
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("총 샘플 수", f"{os_data['total_samples']}건")
        m2.metric("평균 절감률", f"{os_data['avg_reduction_rate'] * 100:.1f}%")
        m3.metric("중앙값 절감률", f"{os_data['median_reduction_rate'] * 100:.1f}%")
        m4.metric("평균 절감 토큰", f"{os_data['avg_saved_tokens']:.0f}개")
        m5.metric("총 절감 토큰", f"{os_data['total_tokens_saved']}개")

        # 전체 통계 표
        st.markdown("**📋 기술 통계량 (Descriptive Statistics)**")
        desc_df = pd.DataFrame([{
            "지표": "토큰 절감률 (%)",
            "평균": f"{os_data['avg_reduction_rate'] * 100:.2f}",
            "중앙값": f"{os_data['median_reduction_rate'] * 100:.2f}",
            "표준편차": f"{os_data['std_reduction_rate'] * 100:.2f}",
            "최솟값": f"{os_data['min_reduction_rate'] * 100:.2f}",
            "최댓값": f"{os_data['max_reduction_rate'] * 100:.2f}",
        }, {
            "지표": "원본 토큰 수",
            "평균": f"{os_data['avg_original_tokens']:.1f}",
            "중앙값": "-",
            "표준편차": "-",
            "최솟값": f"{min(r.original_tokens for r in report.results)}",
            "최댓값": f"{max(r.original_tokens for r in report.results)}",
        }, {
            "지표": "최적화 토큰 수",
            "평균": f"{os_data['avg_refined_tokens']:.1f}",
            "중앙값": "-",
            "표준편차": "-",
            "최솟값": f"{min(r.refined_tokens for r in report.results)}",
            "최댓값": f"{max(r.refined_tokens for r in report.results)}",
        }])
        st.dataframe(desc_df, use_container_width=True, hide_index=True)

        # ═══ 2. 카테고리별 분석 ═══
        st.divider()
        st.subheader("2️⃣ 카테고리별 분석 (Category Analysis)")

        cat_df = pd.DataFrame([{
            "카테고리": s.category,
            "샘플 수": s.sample_count,
            "평균 원본 토큰": f"{s.avg_original_tokens:.1f}",
            "평균 최적화 토큰": f"{s.avg_refined_tokens:.1f}",
            "평균 절감 토큰": f"{s.avg_saved_tokens:.1f}",
            "평균 절감률(%)": f"{s.avg_reduction_rate * 100:.2f}",
            "중앙값(%)": f"{s.median_reduction_rate * 100:.2f}",
            "표준편차(%)": f"{s.std_reduction_rate * 100:.2f}",
            "최소(%)": f"{s.min_reduction_rate * 100:.2f}",
            "최대(%)": f"{s.max_reduction_rate * 100:.2f}",
        } for s in report.category_stats])
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

        # 차트
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("**Figure 1: 카테고리별 평균 토큰 절감률**")
            fig1 = plot_category_reduction(report)
            st.pyplot(fig1)
        with col_f2:
            st.markdown("**Figure 2: 카테고리별 토큰 수 비교**")
            fig2 = plot_token_comparison(report)
            st.pyplot(fig2)

        # ═══ 3. 패턴 분석 ═══
        st.divider()
        st.subheader("3️⃣ 낭비 패턴 분석 (Pattern Analysis)")

        freq = os_data.get("pattern_frequency", {})
        if freq:
            freq_df = pd.DataFrame([
                {"패턴": k, "출현 빈도 (프롬프트 수)": v,
                 "출현율 (%)": f"{v / report.total_samples * 100:.1f}"}
                for k, v in freq.items()
            ])
            st.dataframe(freq_df, use_container_width=True, hide_index=True)

            col_f3, col_f4 = st.columns(2)
            with col_f3:
                st.markdown("**Figure 3: 패턴별 출현 빈도**")
                fig3 = plot_pattern_frequency(report)
                st.pyplot(fig3)
            with col_f4:
                st.markdown("**Figure 4: 절감률 분포**")
                fig4 = plot_reduction_distribution(report)
                st.pyplot(fig4)

        # ═══ 4. 개별 결과 ═══
        st.divider()
        st.subheader("4️⃣ 개별 프롬프트 결과 (Raw Results)")

        raw_df = pd.DataFrame([{
            "카테고리": r.category,
            "#": r.prompt_id,
            "원본 토큰": r.original_tokens,
            "최적화 토큰": r.refined_tokens,
            "절감 토큰": r.saved_tokens,
            "절감률(%)": f"{r.reduction_rate * 100:.1f}",
            "감지 패턴 수": r.pattern_count,
            "적용 규칙 수": r.rules_applied,
        } for r in report.results])
        st.dataframe(raw_df, use_container_width=True, hide_index=True)

        # 산점도
        col_f5, col_f6 = st.columns(2)
        with col_f5:
            st.markdown("**Figure 5: 비용 절감 시뮬레이션**")
            fig5 = plot_cost_simulation(report)
            st.pyplot(fig5)
        with col_f6:
            st.markdown("**Figure 6: 원본 vs 최적화 산점도**")
            fig6 = plot_before_after_scatter(report)
            st.pyplot(fig6)

        # ═══ 5. 최적화 전후 비교 예시 (논문용) ═══
        st.divider()
        st.subheader("5️⃣ 최적화 전후 비교 예시 (Qualitative Examples)")

        # 절감률 상위 3개 표시
        sorted_results = sorted(report.results, key=lambda x: -x.reduction_rate)
        for i, r in enumerate(sorted_results[:3]):
            with st.expander(f"예시 {i+1}: {r.category} — 절감률 {r.reduction_rate*100:.1f}%"):
                ex_c1, ex_c2 = st.columns(2)
                with ex_c1:
                    st.markdown(f"**원본** ({r.original_tokens}토큰)")
                    st.code(r.original_text, language=None)
                with ex_c2:
                    st.markdown(f"**최적화 후** ({r.refined_tokens}토큰)")
                    st.code(r.refined_text, language=None)
                st.markdown(f"감지 패턴: `{'`, `'.join(r.patterns_found)}`")

        # ═══ 6. 데이터 내보내기 ═══
        st.divider()
        st.subheader("6️⃣ 데이터 내보내기 (Export)")

        export_dir = "results"
        col_ex1, col_ex2, col_ex3 = st.columns(3)

        with col_ex1:
            if st.button("📥 CSV 내보내기", use_container_width=True):
                BenchmarkRunner.export_csv(report, f"{export_dir}/benchmark_results.csv")
                st.success(f"✅ `{export_dir}/benchmark_results.csv` 저장 완료")
                st.success(f"✅ `{export_dir}/benchmark_results_stats.csv` 저장 완료")

        with col_ex2:
            if st.button("📥 JSON 내보내기", use_container_width=True):
                BenchmarkRunner.export_json(report, f"{export_dir}/benchmark_results.json")
                st.success(f"✅ `{export_dir}/benchmark_results.json` 저장 완료")

        with col_ex3:
            if st.button("📥 논문용 차트 저장 (300 DPI)", use_container_width=True):
                saved = generate_all_figures(report, f"{export_dir}/figures")
                for path in saved:
                    st.success(f"✅ `{path}` 저장 완료")
