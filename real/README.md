# The real data

Cohesion on the Hot Mess paper's public GPQA runs, to see if [the toy](../toy/README.md) shows up in real models.

- **In this pilot, it seems to.** Real questions with the same error incoherence can have very different Cohesion. Some split into a few clear camps and others are spread out.
- **This is a pilot.** It covers one model at one setting, Claude Sonnet 4 with a 1k thinking budget. It shows there is something to look into, not what drives it.

## Three questions with the same score

![Three panels of 40 runs each, from real GPQA questions. The first has four loose camps of soft answers and Cohesion 0.19. The second has two camps of mostly certain answers and Cohesion 0.76. The third has two camps of fully certain answers and Cohesion 0.95. All three have an error incoherence between 0.50 and 0.57.](figure.png)

These are three real questions from Sonnet 4 at 1k. Each has an error incoherence between 0.50 and 0.57.

- **On the left, Cohesion is 0.19.** The runs split across all four options, and most of them hedge.
- **In the middle, Cohesion is 0.76.** Nearly every run picks A or B and is sure of it.
- **On the right, Cohesion is 0.95.** Every run is 100% sure, with 14 on A and 26 on D. This looks like the toy's two camps.

## It isn't just three questions

In 337 of the 448 questions, the runs disagree. We group those questions by error incoherence and look at the spread of Cohesion in each group.

| Error incoherence | Questions | Cohesion, 10th to 90th percentile |
|---|---|---|
| 0.0 to 0.2 | 44 | 0.40 to 0.95 |
| 0.2 to 0.4 | 33 | 0.28 to 0.95 |
| 0.4 to 0.6 | 30 | 0.27 to 0.95 |
| 0.6 to 0.8 | 21 | 0.25 to 0.91 |
| 0.8 to 1.0 | 209 | 0.54 to 0.97 |

- **Every group spans a wide range of Cohesion.** The narrowest still runs from 0.54 to 0.97.
- **The two measures pick up different things.** Error incoherence gives the share of error that is variance, and Cohesion asks whether that variance falls into camps. So Cohesion may be a useful second number next to error incoherence, as the toy suggests.
- **Reading error incoherence per question is our choice, not the paper's.** The paper pools it over a question set. Per question, it piles up near 1.0, see [open questions](#open-questions).

## The data

The paper's runs are public at [hot-mess/hot-mess-data](https://huggingface.co/datasets/hot-mess/hot-mess-data). We use `claude-sonnet-4-gpqa-prob-fewshot-cot-1k-2`, which has 40 runs of the 448 GPQA questions.

- **Each answer is four probabilities.** The model ends with a line like `<PROB>0.70, 0.05, 0.25, 0.00</PROB>`, so each question gives a 40 × 4 table, the same shape as the toy.
- **Parsing follows the paper's loader.** That is `load_model_data` in [haeggee/hot-mess-of-ai](https://github.com/haeggee/hot-mess-of-ai/blob/main/scripts/utils.py). An answer with no usable probabilities becomes uniform, as in the paper. That happened 7 times in 17,920 answers.
- **The measures are the toy's, with the paper's smoothing.** Before taking logs we add 10⁻¹² to every option, as the paper does. The code lives in [`measures.py`](../measures.py), and the toy uses the same file.
- **We keep scores, not questions.** The paper's runs are MIT licensed, so please cite the paper as its [dataset card](https://huggingface.co/datasets/hot-mess/hot-mess-data) asks. GPQA asks that its questions not be posted online, so `real/results/` holds only per-question scores, with no question text, options or model answers.

## Run it

You need [uv](https://docs.astral.sh/uv/). The first run downloads about 550 MB into `data/`, which is gitignored.

| What | Command |
|---|---|
| Print the table above | `uv run real/main.py` |
| Redraw the figure to `real/figure.png` | `uv run real/plot.py` |
| Run another config from the dataset | `uv run real/main.py --config o4-mini-gpqa-prob-fewshot-cot-high-2` |

Each run writes one row per question to `real/results/<config>.csv`. Parsed answers are cached in `data/cache/`, so a second run takes seconds.

## Open questions

These are worth a look next. We haven't tested any of them.

- **How much does certainty drive Cohesion?** When every run puts 100% on one option, Cohesion reads about 0.95 whatever the split, as [the toy](../toy/README.md#limitations) notes. Some of the spread above may come from that.
- **Does it hold for other models and settings?** This is one model, one thinking budget and one benchmark.
- **Does Cohesion change with longer reasoning?** The paper's main claim is about how incoherence grows with reasoning length. The same question applies to Cohesion.
- **Per-question error incoherence piles up near 1.0.** When most runs are right and sure, the geometric mean puts almost all its weight on the right answer, so bias is near zero and any disagreement scores near 1. The paper pools over questions, so its numbers don't have this problem.
- **Cohesion still ignores the right answer.** As in the toy, two camps of wrong answers can score the same as one right camp and one wrong camp.
