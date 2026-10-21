"""prompt 池：每个 prompt 都带一组**可验证判据**。

这是整个项目的地基。`rule` 裁判唯一的依据就是这里的判据，而 `rule` 裁判是全套
偏差检测的**锚点** —— 如果判据本身有主观成分，「无偏」就没有定义，后面所有的
偏差数字都会变成自说自话。

所以四条判据全部是机械可判定的：

- `exact` 核心答案是否正确（解析出答案后逐字符比对）
- `clean` 有没有「〔?〕」这种犹豫标记
- `punct` 是否以「。」收尾
- `tidy` 有没有多余的双空格

**刻意没有长度判据。** 长度必须是与真值无关的噪声，否则「长度偏差」永远和
「长答案确实更好」纠缠不清 —— 那是真实偏好数据里最难拆的一对混淆，这里把它
拆开才能谈检测。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .rng import Rng, derive_seed
from .schema import PROMPT_KINDS, Check, Prompt, SchemaError

__all__ = [
    "HEDGE",
    "SIGNOFF",
    "TERMINATOR",
    "PREAMBLES",
    "BASE_DIFFICULTY",
    "build_pool",
    "load_prompts",
    "write_pool",
    "render",
    "extract_answer",
    "evaluate",
    "partial_scores",
    "criteria_score",
    "checks_of",
]

#: 犹豫标记（`clean` 判据的抓手）。短，是为了让「违反判据」带来的长度变化
#: 远小于客套话带来的长度变化 —— 长度偏差要能被识别，长度就必须主要由
#: **与真值无关**的填充决定。
HEDGE = "〔?〕"
SIGNOFF = "，希望对你有帮助"
TERMINATOR = "。"

#: 引导语的分块。**按块拼接**是为了让前缀长度落在自然边界上（0/4/7/10/13/17），
#: 而不是随手截断出半句话。
_PRE_CHUNKS: tuple[str, ...] = ("好的，", "这道题", "我仔细", "看了一遍，")
_ANS_MARK: str = "答案是："
#: 落款。只在长度预算够的时候才出现。
SIGNOFF = "，希望对你有帮助"

#: 每条判据的基准难度（会按 prompt 加确定性抖动）。
#:
#: 这一组值是**调过两轮**的：
#:
#: - 第一版 0.34~0.58：576 个候选里 500 个满分，真值退化成几乎二值 —— 76% 的记录
#:   「两个候选一样好」，过滤器的价值就测不出来了（它只要把平局全丢掉就很漂亮）。
#: - 第二版 0.55~0.80：好一些，但仍有 46% 拿满分，真值还是挤在顶端。
#: - 现在把难度推到候选能力**之上**，于是「完全答对」是少数，`exact` 的部分分
#:   才是主要区分度来源 —— 真值这才铺开。
BASE_DIFFICULTY: dict[str, float] = {
    "exact": 0.90,
    "clean": 0.70,
    "punct": 0.62,
    "tidy": 0.80,
}

_PHRASES: tuple[str, ...] = (
    "hello world", "deep learning", "token factory", "batch norm",
    "cross entropy", "attention head", "low rank", "kv cache",
    "mixed precision", "rope scaling", "sparse moe", "reward model",
    "preference pair", "judge pool", "distill loss", "context window",
)
_WORDS: tuple[str, ...] = (
    "abcdef", "gravity", "monsoon", "bracket", "quantum", "lantern",
    "prismal", "vectors", "octagon", "saffron", "tundra", "zeppelin",
    "cascade", "marbled", "wanderer", "harbor",
)


# --------------------------------------------------------------------------
# 池的构造
# --------------------------------------------------------------------------

def _make_arith(rng: Rng, i: int) -> tuple[str, str]:
    if rng.bernoulli(0.5):
        a, b = rng.randint(11, 89), rng.randint(11, 89)
        return f"计算 {a} + {b}，只给数字。", str(a + b)
    a, b = rng.randint(3, 12), rng.randint(3, 9)
    return f"计算 {a} × {b}，只给数字。", str(a * b)


def _make_upper(rng: Rng, i: int) -> tuple[str, str]:
    s = _PHRASES[(i + rng.randint(0, len(_PHRASES) - 1)) % len(_PHRASES)]
    return f"把「{s}」全部改成大写，只给结果。", s.upper()


def _make_reverse(rng: Rng, i: int) -> tuple[str, str]:
    s = _WORDS[(i + rng.randint(0, len(_WORDS) - 1)) % len(_WORDS)]
    return f"把「{s}」倒过来写，只给结果。", s[::-1]


def _make_sort(rng: Rng, i: int) -> tuple[str, str]:
    nums = rng.sample(tuple(range(1, 10)), 4)
    shown = "/".join(str(n) for n in nums) if rng.bernoulli(0.5) else "/".join(str(n) for n in rng.shuffled(nums))
    return f"把 {shown} 从小到大排好，用 / 分隔。", "/".join(str(n) for n in sorted(nums))


_MAKERS = {"arith": _make_arith, "upper": _make_upper, "reverse": _make_reverse, "sort": _make_sort}


def build_pool(rng: Rng, n: int) -> list[Prompt]:
    """构造 `n` 个 prompt。**同一个 rng 调用序列 → 同一个池**，于是版本可复现。

    `kind` 轮转而不是随机：四类题型的样本量必须均衡，否则「裁判一致率」会被
    题型分布污染（比如全是算术时 rule 裁判的区分度会显得特别高）。
    """
    if n <= 0:
        raise SchemaError(f"prompt 池大小必须 > 0，得到 {n}")
    out: list[Prompt] = []
    for i in range(n):
        kind = PROMPT_KINDS[i % len(PROMPT_KINDS)]
        question, expected = _MAKERS[kind](rng, i)
        checks = tuple(
            Check(k, _jitter(rng, BASE_DIFFICULTY[k], f"{i}:{k}"))
            for k in ("exact", "clean", "punct", "tidy")
        )
        out.append(Prompt(id=f"p{i:04d}", kind=kind, question=question, expected=expected, checks=checks))
    return out


def _jitter(rng: Rng, base: float, tag: str) -> float:
    d = base + rng.uniform(-0.05, 0.05)
    return round(min(0.95, max(0.05, d)), 6)


def write_pool(path: Path | str, prompts: Iterable[Prompt]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for pr in prompts:
            fh.write(json.dumps(pr.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def load_prompts(path: Path | str) -> list[Prompt]:
    p = Path(path)
    if not p.exists():
        raise SchemaError(f"prompt 池不存在：{p}")
    out: list[Prompt] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Prompt.from_dict(json.loads(line)))
    if not out:
        raise SchemaError(f"prompt 池是空的：{p}")
    return out


# --------------------------------------------------------------------------
# 文本渲染
# --------------------------------------------------------------------------

def _preamble_for(n: int) -> str:
    """返回长度 **<= n** 的最长自然引导语（以「答案是：」收尾）。"""
    if n < len(_ANS_MARK):
        return _ANS_MARK[: max(0, n)]
    best = _ANS_MARK
    acc = ""
    for ch in _PRE_CHUNKS:
        cand = acc + ch + _ANS_MARK
        if len(cand) <= n:
            best = cand
            acc += ch
        else:
            break
    return best


def _arith_scale(expected_int: int) -> float:
    """算术误差的归一化尺度。**扰动与打分必须用同一个尺度**：
    否则部分分还原不出「错得有多远」，真值又会退化成分级的。
    """
    return max(3.0, abs(expected_int) / 4.0)


def _hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x != y)


def _perturb(expected: str, kind: str, rng: Rng, err: float) -> str:
    """把正确答案改成一个**像样但错误**的答案，**错误幅度 ∝ err**。

    两个要点：

    - 「像样」很重要：如果错误答案一眼假，裁判（哪怕是噪声很大的那个）都会判对，
      于是所有裁判的一致率都接近 1，过滤器的价值就测不出来了。
    - **错误幅度必须与 `err`（离及格线有多远）挂钩。** 第一版这里没挂钩，
      于是「错一点」和「错很远」拿到同一个部分分，真值退化成了分级的 ——
      实测 576 个候选里 500 个满分，76% 的记录变成「两个候选一样好」，
      过滤器只要把平局全丢掉就能拿到漂亮数字。这是本项目踩的最大的一个坑。
    """
    e = min(1.0, max(0.05, err))
    if kind == "arith":
        delta = max(1, int(round(e * _arith_scale(int(expected)))))
        val = int(expected) + (delta if rng.bernoulli(0.5) else -delta)
        return str(max(0, val))
    chars = list(expected)
    n = len(chars)
    if n < 2:
        return expected + "x"
    if kind == "upper":
        # 只挑**大写字母**去小写化。挑到空格上就等于没改 —— 那会让「标记成违反
        # exact」的候选重新解析回来仍然是满分，渲染 ↔ 判定的往返就断了
        # （实测 288 个候选里有 4 个栽在这上面）。
        idx = [j for j in range(n) if chars[j].isupper()]
        if not idx:
            return expected + "x"
        for j in rng.sample(tuple(idx), max(1, min(len(idx), int(round(e * n))))):
            chars[j] = chars[j].lower()
        return "".join(chars)
    if kind == "reverse":
        # 只换**内容不同**的两位：字符串里有重复字母时，交换两个相同的字符等于
        # 没改，同样会断掉往返。
        movable = [(i, j) for i in range(n) for j in range(i + 1, n) if chars[i] != chars[j]]
        if not movable:
            return expected + "x"
        for _ in range(max(1, min(len(movable), int(round(e * n / 2))))):
            i, j = movable[rng.randint(0, len(movable) - 1)]
            chars[i], chars[j] = chars[j], chars[i]
        return "".join(chars)
    # sort：交换相邻若干对，看起来「差点就对了」。
    parts = expected.split("/")
    if len(parts) >= 2:
        for _ in range(max(1, min(len(parts) - 1, int(round(e * len(parts) / 2))))):
            i = rng.randint(0, len(parts) - 2)
            parts[i], parts[i + 1] = parts[i + 1], parts[i]
        return "/".join(parts)
    return expected


def render(
    prompt: Prompt,
    violated: frozenset[str],
    verbosity: float,
    rng: Rng,
    err: float = 0.5,
) -> tuple[str, int]:
    """渲染候选文本。返回 `(文本, 与真值无关的填充字符数)`。

    组成顺序固定：`引导语 + 犹豫标记? + 答案 + 落款? + 多余空格? + 句号?`
    顺序固定是 `extract_answer` 能机械解析的前提 —— 渲染与解析必须是一对，
    否则 `rule` 裁判看到的就不是候选真实写的东西。

    **长度预算制**：`verbosity` 先定一个总填充预算，犹豫标记 / 多余空格 / 句号
    这些「违反判据的代价」**先从预算里扣**，剩下的才用来铺引导语。

    这条是必须的。第一版让违反判据直接加字符，于是「答得差」和「写得长」绑在一起，
    实测 corr(长度, 真值) = **−0.361** —— 长度偏差和「长答案确实更差」彻底纠缠，
    长度轴的检出率就没有干净的解释了。改成预算制之后，长度主要由 `verbosity`
    决定，而与真值无关。
    """
    core = _perturb(prompt.expected, prompt.kind, rng, err) if "exact" in violated else prompt.expected
    hedge = HEDGE if "clean" in violated else ""
    noise = "  " if "tidy" in violated else ""
    term = "" if "punct" in violated else TERMINATOR
    budget = max(0, int(round(verbosity * 10.0)))
    tail = SIGNOFF if budget >= 14 else ""
    fixed = len(hedge) + len(noise) + len(term) + len(tail)
    pre = _preamble_for(budget - fixed)
    text = f"{pre}{hedge}{core}{tail}{noise}{term}"
    pad = len(pre) + fixed
    return text, pad


# --------------------------------------------------------------------------
# 判据求值（rule 裁判的全部依据）
# --------------------------------------------------------------------------

def extract_answer(prompt: Prompt, text: str) -> str:
    """从候选文本里机械地抽出「核心答案」。

    三步剥离：句号 → 落款 → 客套话（最后一个「：」之前的部分）→ 犹豫标记。
    这是**渲染顺序的逆**，所以它只能解析本项目渲染出来的文本 —— 这正是我们要的：
    `rule` 裁判不是通用语言理解器，它是一台针对这批候选的判定机。
    """
    s = text
    if s.endswith(TERMINATOR):
        s = s[: -len(TERMINATOR)]
    if s.endswith(SIGNOFF):
        s = s[: -len(SIGNOFF)]
    if "：" in s:
        s = s.rsplit("：", 1)[1]
    if s.startswith(HEDGE):
        s = s[len(HEDGE) :]
    return s.strip()


def evaluate(prompt: Prompt, text: str) -> dict[str, bool]:
    """对一条候选文本判定四条判据。**纯函数、无随机、无长度项。**"""
    return {
        "exact": extract_answer(prompt, text) == prompt.expected,
        "clean": HEDGE not in text,
        "punct": text.endswith(TERMINATOR),
        "tidy": "  " not in text,
    }


def partial_scores(prompt: Prompt, text: str) -> dict[str, float]:
    """四条判据的**部分分**（都在 [0, 1]）。

    二值的「过/不过」会让真值退化成 5 级阶梯，于是大量记录变成「两个候选一样好」。
    部分分让 `exact` 变成连续的：错得越远分越低。而且它**仍然完全机械** ——
    `rule` 裁判的无偏性一点没丢，丢的只是粗糙度。
    """
    ans = extract_answer(prompt, text)
    return {
        "exact": _partial_exact(prompt, ans),
        "clean": 0.0 if HEDGE in text else 1.0,
        "punct": 1.0 if text.endswith(TERMINATOR) else 0.0,
        "tidy": 0.0 if "  " in text else 1.0,
    }


def _partial_exact(prompt: Prompt, ans: str) -> float:
    exp = prompt.expected
    if prompt.kind == "arith":
        try:
            v = int(ans)
        except ValueError:
            return 0.0
        e = abs(v - int(exp))
        return max(0.0, 1.0 - e / _arith_scale(int(exp)))
    n = max(1, len(exp))
    return max(0.0, 1.0 - _hamming(exp, ans) / n)


def criteria_score(prompt: Prompt, text: str) -> float:
    """`rule` 裁判眼里的客观质量 = 四条判据部分分的均值 ∈ [0, 1]。

    **这是可观测的那部分真值。** 真值里还有 20% 是任何机械判据都看不到的
    （见 `candidates.generate` 里的 `polish`）—— 没有这个残差，`rule` 裁判就
    变成了神谕，「rule 同意才保留」这条过滤规则会强得不真实。
    """
    ps = partial_scores(prompt, text)
    return sum(ps.values()) / len(ps)


def checks_of(prompt: Prompt, text: str) -> tuple[str, ...]:
    """通过的判据列表（按 `CHECK_KINDS` 的顺序，便于做集合比较）。"""
    res = evaluate(prompt, text)
    return tuple(k for k in ("exact", "clean", "punct", "tidy") if res.get(k))
