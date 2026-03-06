"""
LLM Client
==========
OpenAI API를 직접 호출하여 In-Context Learning 
기반 프롬프트 최적화를 수행하는 모듈.
"""

import g4f
from g4f.client import Client

class LLMOptimizerClient:
    """
    g4f (GPT4Free) API를 활용하여 시스템 프롬프트(도메인 지침)와
    Few-Shot(RAG 검색 사례)를 바탕으로 프롬프트를 최적화한다.
    """
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        # g4f 모델은 보통 gpt-4, gpt-3.5-turbo 등으로 매핑됨.
        self.model = "gpt-4o-mini" if model == "gpt-4o-mini" else "gpt-3.5-turbo"
        # OpenAI API Key는 필요 없음
        self.client = Client()

    def optimize_prompt(
        self,
        target_prompt: str,
        domain: str,
        rag_examples: list[dict]
    ) -> str:
        """
        주어진 프롬프트를 AI 모델(무료)을 통해 최적화한다.
        """
        system_prompt = self._build_system_prompt(domain)
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Few-Shot Examples (요청과 최적화 결과 쌍)
        for ex in rag_examples:
            messages.append({"role": "user", "content": f"다음 프롬프트를 최적화해 줘:\n{ex['original']}"})
            messages.append({"role": "assistant", "content": ex['refined']})

        # 타겟 프롬프트 추가
        messages.append({"role": "user", "content": f"다음 프롬프트를 최적화해 줘:\n{target_prompt}"})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=600,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"g4f Error: {e}")
            raise RuntimeError(f"무료 LLM 생성 중 에러가 발생했습니다: {e}")

    def _build_system_prompt(self, domain: str) -> str:
        """도메인별 특화된 커스텀 지침(페르소나) 생성"""
        
        base_instruction = (
            "당신은 AI 프롬프트 최적화 전문가(Prompt Optimizer)입니다.\n"
            "사용자의 원본 프롬프트를 분석하여 가장 중요한 핵심 요지와 의도를 100% 유지하면서, "
            "불필요한 인사말, 장황한 설명, 중복되는 표현을 제거하여 **가장 짧은 토큰(글자) 수**로 작성해 주세요.\n"
            "결과물은 어색하지 않은 명확한 지시문이어야 합니다. 요약된 프롬프트 내용만 답변하고 다른 설명은 절대 추가하지 마세요.\n\n"
        )

        domain_instruction = {
            "질문응답": "추가 지침: '자세히 설명해주세요', '알려주세요' 같은 메타 요청을 최소화하고, 질문 자체를 직관적으로 간결하게 만드세요.",
            "코드생성": "추가 지침: '코드 작성 부탁드립니다' 같은 요청을 제거하고, 기술 스택, 요구사항, 구현 목표만 명사형 또는 간결한 문장으로 나열하세요.",
            "요약": "추가 지침: '아래 글을 읽고' 같은 불필요한 맥락 지시를 지우고, '요약'과 형식(예: '3줄')만 명확히 강조하세요.",
            "번역": "추가 지침: '자연스럽게 번역해줘' 등을 '자연스럽게 번역' 등으로 축약하고, 원본 텍스트는 그대로 유지하되 주변 지시어만 최적화하세요."
        }

        specific = domain_instruction.get(domain, "핵심 의도를 파악하여 직관적이고 군더더기 없이 최적화하세요.")
        
        return base_instruction + specific
