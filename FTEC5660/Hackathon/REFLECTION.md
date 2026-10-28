# Self-Reflection

Name：Yiwei Wan

Student Number：1155261963

Hackathon：[AI Safety Hackathon](https://www.aisafetyhk.org/apart-hackathon-hk-hub)

Project：[Communication Limits, Oversight, and Profit Pressure in LLM Double Auctions](https://apartresearch.com/sprints/projects/communication-limits-oversight-and-profit-pressure-in-llm-double-auctions-hux1)

## What is my project



## What worked well

My project examined whether communication limits and oversight could reduce coordination among AI sellers. I adapted the existing double-auction simulator, refined its multi-agent workflow and data structures, and cleaned up outdated code and files. The agents could make decisions in parallel, exchange messages between rounds, and retain their own memories and strategy notes. I also reorganized the experiment records so I could trace results back to model calls, messages, and trading decisions.

I integrated Typesafe Jev to assess seller messages and trigger a warning in the following round when its coordination score crossed a threshold. This allowed me to compare oversight with communication limits and profit pressure across seven conditions. Looking at both seller messages and subsequent quotes helped me make a more grounded assessment of coordination.

## What worked poorly

Each condition used only one random seed, so I could not separate the observed differences from random variation. The findings were useful observations, but I could not assume that the same patterns would hold across repeated runs.

I also did not initially handle all invalid model responses in the retry mechanism. For example, a model could omit a required field when it intended the field to be empty. I added stricter validation afterward, but I should have identified this case before the formal experiments.

Finally, I used Jev's scoring criteria after setting them up without calibrating them against independent annotations. Some messages may have been misclassified, so its scores should not be treated as conclusive evidence of coordination.

## What I would improve

I would run each condition with multiple seeds and include a wider range of comparison groups. I would test actual financial penalties in the simulation, rather than relying only on written warnings. I would also explore stronger ways to model pressure from a boss, such as imposing a minimum seller profit requirement, and observe how this affects both coordination and trade.

The original paper used GPT-4.1, while I used a newer model. I would revisit the prompts for the model used and explore more complex trading scenarios. I would also test whether Jev could help evaluate or select actions for individual buyers and sellers, in addition to monitoring seller messages. Before relying on its scores, I would compare them with independent human annotations.

## What I have learned

This was my first hackathon. Working under a tight deadline showed me how much could be accomplished in a short time when I focused on the most important work. It also taught me that producing a working experiment is only part of the task; its design and results need careful scrutiny.

This was also my first direct experience with AI safety. Through the project, I gained an initial understanding of how risks can emerge from interactions among AI agents, and why assessing those risks requires looking beyond individual model responses.
