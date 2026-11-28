# error-cohesion

A [toy example](toy/README.md) and a [pilot on real data](real/README.md) that build on the paper [Hot Mess](https://arxiv.org/abs/2601.23045) as a possible follow-up.

From the paper's abstract:

> AI can fail because it is misaligned, and produces consistent but undesired outcomes, or because it is incoherent, and does not produce consistent outcomes at all. These failures correspond to bias and variance respectively. As we extrapolate risks from AI, it is important to understand whether failures from more capable models performing more complex tasks will be bias or variance dominated. Bias dominated failures will look like model misalignment, while variance dominated failures will resemble industrial accidents.

Error incoherence measures how much of a model's error comes from variance. By design, it doesn't say how that variance is arranged. The answers could be spread at random, or they could fall into a few organised camps. [Cohesion](https://timjdavey.com/cohesion/) ([paper](https://doi.org/10.3390/e25121605) + [package](https://github.com/timjdavey/incoherence)) is one way to measure that second question, alongside error incoherence.

![Three panels of 30 samples each. The first has a different mix in every row, the second has five camps and the third has two camps. All three have the same error incoherence.](toy/figure.png)

In terms of the framing above, high error incoherence with high Cohesion may look more like a few steady behaviours than a hot mess. High error incoherence with low Cohesion is closest to the random spread the name suggests. This seems a distinction worth testing.

I'd be interested to see how this plays out on longer task horizons. In some cases the model may fall into a few "attractor" states (from chaos terminology), rather than spreading evenly.

A similar pattern shows up in the paper's public GPQA runs, see [the real data](real/README.md). It might also lead to better constructions of Cohesion.

![Three panels of 40 runs each, from real GPQA questions. The first has four loose camps of soft answers and Cohesion 0.19. The second has two camps of mostly certain answers and Cohesion 0.76. The third has two camps of fully certain answers and Cohesion 0.95. All three have an error incoherence between 0.50 and 0.57.](real/figure.png)

Error incoherence shares a name with a different measure, [Incoherence](https://doi.org/10.3390/e26080683). They are built for different jobs, see [how they differ](compare/README.md).

Thanks to the Hot Mess authors for publishing their runs and their loading code. The pilot would not be possible without both.
