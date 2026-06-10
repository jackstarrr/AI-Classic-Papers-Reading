#!/usr/bin/env python3
"""Small runnable examples for the 30-paper AI reading library."""

from __future__ import annotations

import math
import textwrap
from typing import Callable

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def banner(title: str, body: str) -> str:
    return textwrap.dedent(f"""
    === {title} ===
    {body.strip()}
    """).strip()


def ex01_perceptron() -> str:
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([-1, -1, -1, 1])
    w = np.zeros(2)
    b = 0.0
    updates = 0
    for _ in range(10):
        for xi, yi in zip(x, y):
            if yi * (xi @ w + b) <= 0:
                w += yi * xi
                b += yi
                updates += 1
    pred = np.where(x @ w + b >= 0, 1, -1)
    return banner("Perceptron online learning", f"weights={w.tolist()}, bias={b:.1f}, updates={updates}\npredictions={pred.tolist()}")


def ex02_backprop() -> str:
    rng = np.random.default_rng(2)
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    w1 = rng.normal(0, 1, (2, 4))
    b1 = np.zeros((1, 4))
    w2 = rng.normal(0, 1, (4, 1))
    b2 = np.zeros((1, 1))
    lr = 0.8
    for _ in range(4000):
        h = sigmoid(x @ w1 + b1)
        out = sigmoid(h @ w2 + b2)
        d_out = (out - y) * out * (1 - out)
        d_w2 = h.T @ d_out / len(x)
        d_b2 = d_out.mean(axis=0, keepdims=True)
        d_h = d_out @ w2.T * h * (1 - h)
        d_w1 = x.T @ d_h / len(x)
        d_b1 = d_h.mean(axis=0, keepdims=True)
        w2 -= lr * d_w2
        b2 -= lr * d_b2
        w1 -= lr * d_w1
        b1 -= lr * d_b1
    pred = (out > 0.5).astype(int).ravel().tolist()
    return banner("Backprop on XOR", f"final probabilities={np.round(out.ravel(), 3).tolist()}\npredictions={pred}")


def ex03_lenet() -> str:
    image = np.arange(25, dtype=float).reshape(5, 5)
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)
    conv = np.array([[np.sum(image[i:i + 3, j:j + 3] * kernel) for j in range(3)] for i in range(3)])
    pool = np.array([[conv[:2, :2].max(), conv[:2, 1:].max()], [conv[1:, :2].max(), conv[1:, 1:].max()]])
    return banner("Convolution and max pooling", f"conv map=\n{conv}\npooled map=\n{pool}")


def ex04_lstm() -> str:
    x = np.array([0.7, -0.4])
    h_prev = np.array([0.1, 0.2])
    c_prev = np.array([0.3, -0.1])
    z = np.concatenate([x, h_prev])
    wf = np.array([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5], [0.1, 0.2]])
    wi = np.array([[0.1, 0.4], [-0.3, 0.2], [0.5, -0.2], [0.2, 0.1]])
    wo = np.array([[0.3, -0.2], [0.1, 0.4], [0.2, 0.2], [-0.4, 0.3]])
    wg = np.array([[0.2, 0.1], [0.2, -0.3], [0.1, 0.4], [0.5, 0.2]])
    f, i, o = sigmoid(z @ wf), sigmoid(z @ wi), sigmoid(z @ wo)
    g = np.tanh(z @ wg)
    c = f * c_prev + i * g
    h = o * np.tanh(c)
    return banner("LSTM one step", f"forget={np.round(f, 3).tolist()}, input={np.round(i, 3).tolist()}, output={np.round(o, 3).tolist()}\ncell={np.round(c, 3).tolist()}, hidden={np.round(h, 3).tolist()}")


def ex05_svm() -> str:
    x = np.array([[2, 2], [2, 0], [0, 0], [0, 2]], dtype=float)
    y = np.array([1, 1, -1, -1])
    w = np.array([1.0, 0.0])
    b = -1.0
    margins = y * (x @ w + b)
    hinge = np.maximum(0, 1 - margins)
    support = np.where(np.isclose(margins, 1.0))[0].tolist()
    return banner("SVM margin geometry", f"margins={margins.tolist()}\nhinge losses={hinge.tolist()}\nsupport-vector indices={support}")


def ex06_random_forests() -> str:
    x = np.array([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1], [0.4, 0.6]])
    stumps = [(0, 0.5), (1, 0.5), (0, 0.3)]
    votes = []
    for feature, threshold in stumps:
        votes.append(np.where(x[:, feature] > threshold, 1, 0))
    votes = np.array(votes)
    pred = (votes.mean(axis=0) >= 0.5).astype(int)
    return banner("Random forest voting", f"tree votes=\n{votes}\nmajority prediction={pred.tolist()}")


def ex07_lda() -> str:
    doc_topic = np.array([3, 1])
    topic_word = np.array([[4, 1, 1], [1, 3, 2]])
    alpha, beta = 0.1, 0.1
    word_id = 1
    topic_scores = (doc_topic + alpha) * (topic_word[:, word_id] + beta) / (topic_word.sum(axis=1) + 3 * beta)
    probs = topic_scores / topic_scores.sum()
    return banner("LDA Gibbs conditional", f"topic scores={np.round(topic_scores, 4).tolist()}\nsampling probabilities={np.round(probs, 4).tolist()}")


def ex08_alexnet() -> str:
    def conv_out(size: int, kernel: int, stride: int, pad: int = 0) -> int:
        return (size + 2 * pad - kernel) // stride + 1

    shapes = []
    size = 224
    for name, kernel, stride, pad, channels in [
        ("conv1", 11, 4, 2, 96),
        ("pool1", 3, 2, 0, 96),
        ("conv2", 5, 1, 2, 256),
        ("pool2", 3, 2, 0, 256),
    ]:
        size = conv_out(size, kernel, stride, pad)
        shapes.append((name, channels, size, size))
    return banner("AlexNet-style shape flow", "\n".join(f"{n}: {c}x{h}x{w}" for n, c, h, w in shapes))


def ex09_dropout() -> str:
    rng = np.random.default_rng(9)
    activations = np.ones(10)
    keep_prob = 0.6
    mask = rng.binomial(1, keep_prob, size=activations.shape)
    train = activations * mask / keep_prob
    test = activations
    return banner("Inverted dropout", f"mask={mask.tolist()}\ntrain activations={np.round(train, 2).tolist()}\ntest activations={test.tolist()}")


def ex10_vae() -> str:
    mu = np.array([0.5, -0.2])
    log_var = np.array([-0.7, 0.2])
    eps = np.array([0.1, -1.0])
    z = mu + np.exp(0.5 * log_var) * eps
    kl = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
    return banner("VAE reparameterization", f"z = mu + sigma * eps = {np.round(z, 3).tolist()}\nKL(q(z|x)||p(z))={kl:.4f}")


def ex11_gan() -> str:
    rng = np.random.default_rng(11)
    theta = -1.0
    for _ in range(20):
        real = rng.normal(1.0, 0.2, 32)
        fake = rng.normal(theta, 0.2, 32)
        theta += 0.1 * (real.mean() - fake.mean())
    return banner("GAN toy generator mean", f"target real mean is near 1.0\ngenerator mean after updates={theta:.3f}")


def ex12_word2vec() -> str:
    rng = np.random.default_rng(12)
    vin = rng.normal(0, 0.1, (4, 3))
    vout = rng.normal(0, 0.1, (4, 3))
    center, positive, negative = 0, 1, 2
    lr = 0.5
    h = vin[center].copy()
    grad_h = np.zeros_like(h)
    for word, label in [(positive, 1), (negative, 0)]:
        score = sigmoid(h @ vout[word])
        grad = score - label
        grad_h += grad * vout[word]
        vout[word] -= lr * grad * h
    vin[center] -= lr * grad_h
    return banner("Skip-gram negative sampling", f"updated center vector={np.round(vin[center], 4).tolist()}")


def ex13_seq2seq() -> str:
    source = [1, 2, 3]
    emb = np.eye(5)
    w = np.ones((5, 3)) * 0.2
    h = np.zeros(3)
    for token in source:
        h = np.tanh(emb[token] @ w + 0.5 * h)
    decoder_state = h.copy()
    logits = decoder_state @ np.array([[1.0, -0.5], [0.2, 0.3], [-0.1, 0.8]])
    return banner("Seq2Seq context vector", f"encoder final state={np.round(h, 3).tolist()}\ndecoder first logits={np.round(logits, 3).tolist()}")


def ex14_attention() -> str:
    keys = np.array([[0.2, 0.1], [0.0, 0.5], [0.7, 0.2]])
    query = np.array([0.6, 0.1])
    wa = np.array([1.0, -0.5])
    scores = np.tanh(keys + query) @ wa
    weights = softmax(scores)
    context = weights @ keys
    return banner("Additive attention", f"scores={np.round(scores, 3).tolist()}\nweights={np.round(weights, 3).tolist()}\ncontext={np.round(context, 3).tolist()}")


def ex15_adam() -> str:
    x = 3.0
    m = v = 0.0
    lr, beta1, beta2 = 0.2, 0.9, 0.999
    path = []
    for t in range(1, 8):
        g = 2 * (x - 1)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        x -= lr * (m / (1 - beta1**t)) / (math.sqrt(v / (1 - beta2**t)) + 1e-8)
        path.append(round(x, 3))
    return banner("Adam on a quadratic", f"parameter path={path}\noptimum=1.0")


def ex16_batchnorm() -> str:
    x = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    gamma = np.array([1.5, 0.5])
    beta = np.array([0.0, 1.0])
    normalized = (x - x.mean(axis=0)) / np.sqrt(x.var(axis=0) + 1e-5)
    y = gamma * normalized + beta
    return banner("Batch normalization", f"normalized mean={np.round(normalized.mean(axis=0), 5).tolist()}\noutput=\n{np.round(y, 3)}")


def ex17_dqn() -> str:
    q = np.zeros((2, 2))
    transitions = [(0, 1, 1.0, 1), (1, 0, 0.5, 0), (0, 0, -0.2, 0)]
    alpha, gamma = 0.5, 0.9
    for s, a, r, ns in transitions:
        target = r + gamma * q[ns].max()
        q[s, a] += alpha * (target - q[s, a])
    return banner("Q-learning replay updates", f"Q table after replay=\n{np.round(q, 3)}")


def ex18_resnet() -> str:
    x = np.array([1.0, -1.0, 0.5])
    w = np.array([[0.2, 0.0, 0.1], [0.1, 0.3, -0.2], [0.0, 0.2, 0.2]])
    residual = np.maximum(0, x @ w)
    y = x + residual
    return banner("Residual block", f"x={x.tolist()}\nF(x)={np.round(residual, 3).tolist()}\ny=F(x)+x={np.round(y, 3).tolist()}")


def ex19_unet() -> str:
    encoder = np.array([[1, 2], [3, 4]])
    bottleneck = encoder.mean()
    upsampled = np.full_like(encoder, bottleneck)
    skipped = np.stack([upsampled, encoder], axis=-1)
    return banner("U-Net skip connection", f"upsampled feature=\n{upsampled}\nconcatenated shape={skipped.shape}")


def ex20_alphago() -> str:
    prior = np.array([0.55, 0.25, 0.20])
    value_sum = np.array([8.0, 3.0, 1.0])
    visits = np.array([10, 4, 1])
    total = visits.sum()
    c_puct = 1.5
    q = value_sum / visits
    u = c_puct * prior * math.sqrt(total) / (1 + visits)
    score = q + u
    return banner("MCTS PUCT selection", f"Q={np.round(q, 3).tolist()}\nU={np.round(u, 3).tolist()}\nselected action={int(score.argmax())}")


def ex21_gcn() -> str:
    a = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=float)
    h = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    w = np.array([[1.0, -0.5], [0.5, 1.0]])
    d_inv_sqrt = np.diag(1 / np.sqrt(a.sum(axis=1)))
    h_next = d_inv_sqrt @ a @ d_inv_sqrt @ h @ w
    return banner("Graph convolution", f"normalized graph convolution output=\n{np.round(h_next, 3)}")


def ex22_transformer() -> str:
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    k = np.array([[1.0, 0.0], [0.5, 0.5]])
    v = np.array([[1.0, 2.0], [3.0, 0.0]])
    weights = softmax(q @ k.T / math.sqrt(q.shape[1]), axis=1)
    out = weights @ v
    return banner("Scaled dot-product attention", f"attention weights=\n{np.round(weights, 3)}\noutput=\n{np.round(out, 3)}")


def ex23_bert() -> str:
    tokens = ["[CLS]", "deep", "learning", "changed", "ai", "[SEP]"]
    masked = tokens.copy()
    masked[2] = "[MASK]"
    labels = {2: "learning"}
    return banner("Masked language modeling sample", f"input tokens={masked}\nlabels={labels}")


def ex24_scaling_laws() -> str:
    params = np.array([1, 3, 10, 30, 100], dtype=float)
    loss = 2.0 * params ** -0.08 + 0.6
    slope, intercept = np.polyfit(np.log(params), np.log(loss - 0.6), 1)
    return banner("Toy scaling-law fit", f"fitted exponent={slope:.3f}\nlosses={np.round(loss, 3).tolist()}")


def ex25_gpt3() -> str:
    demos = {"great": "positive", "awful": "negative", "useful": "positive"}
    test = "awful but useful"
    scores = {label: 0 for label in set(demos.values())}
    for word, label in demos.items():
        if word in test:
            scores[label] += 1
    pred = max(scores, key=scores.get)
    return banner("Few-shot prompt-style classifier", f"demos={demos}\ntest='{test}'\nscores={scores}, prediction={pred}")


def ex26_vit() -> str:
    image = np.arange(4 * 4).reshape(4, 4)
    patches = image.reshape(2, 2, 2, 2).swapaxes(1, 2).reshape(4, 4)
    positions = np.arange(len(patches))[:, None]
    embeddings = patches + positions
    return banner("ViT patchify", f"patches=\n{patches}\nposition-added embeddings=\n{embeddings}")


def ex27_ddpm() -> str:
    x0 = np.array([1.0, -1.0])
    eps = np.array([0.3, -0.7])
    alpha_bar = 0.6
    xt = math.sqrt(alpha_bar) * x0 + math.sqrt(1 - alpha_bar) * eps
    return banner("Diffusion forward noising", f"x0={x0.tolist()}\nepsilon={eps.tolist()}\nx_t={np.round(xt, 3).tolist()}")


def ex28_clip() -> str:
    image_emb = np.array([[1.0, 0.1], [0.2, 0.9]])
    text_emb = np.array([[0.9, 0.0], [0.0, 1.0]])
    image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)
    text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
    logits = image_emb @ text_emb.T / 0.07
    return banner("CLIP contrastive logits", f"image-text logits=\n{np.round(logits, 2)}\nmatched text indices={logits.argmax(axis=1).tolist()}")


def ex29_alphafold2() -> str:
    pair = np.zeros((3, 3))
    msa_signal = np.array([0.2, 0.8, 0.4])
    pair += msa_signal[:, None] + msa_signal[None, :]
    distance_hint = 1 / (1 + pair)
    return banner("Pair representation intuition", f"pair features=\n{np.round(pair, 2)}\ndistance hint=\n{np.round(distance_hint, 2)}")


def ex30_instructgpt() -> str:
    reward_chosen = np.array([1.2, 0.7, 1.5])
    reward_rejected = np.array([0.2, 1.0, 0.4])
    diff = reward_chosen - reward_rejected
    loss = -np.log(sigmoid(diff)).mean()
    accuracy = (diff > 0).mean()
    return banner("Preference ranking loss", f"reward differences={np.round(diff, 3).tolist()}\nloss={loss:.4f}, pairwise accuracy={accuracy:.2f}")


EXAMPLES: dict[str, Callable[[], str]] = {
    "01": ex01_perceptron,
    "02": ex02_backprop,
    "03": ex03_lenet,
    "04": ex04_lstm,
    "05": ex05_svm,
    "06": ex06_random_forests,
    "07": ex07_lda,
    "08": ex08_alexnet,
    "09": ex09_dropout,
    "10": ex10_vae,
    "11": ex11_gan,
    "12": ex12_word2vec,
    "13": ex13_seq2seq,
    "14": ex14_attention,
    "15": ex15_adam,
    "16": ex16_batchnorm,
    "17": ex17_dqn,
    "18": ex18_resnet,
    "19": ex19_unet,
    "20": ex20_alphago,
    "21": ex21_gcn,
    "22": ex22_transformer,
    "23": ex23_bert,
    "24": ex24_scaling_laws,
    "25": ex25_gpt3,
    "26": ex26_vit,
    "27": ex27_ddpm,
    "28": ex28_clip,
    "29": ex29_alphafold2,
    "30": ex30_instructgpt,
}


def run_example(example_id: str) -> str:
    try:
        fn = EXAMPLES[example_id]
    except KeyError as exc:
        raise SystemExit(f"Unknown example id: {example_id}") from exc
    return fn()


if __name__ == "__main__":
    for key in sorted(EXAMPLES):
        print(run_example(key))
        print()

