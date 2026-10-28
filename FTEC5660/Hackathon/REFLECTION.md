# Self-Reflection

Name：Yiwei Wan			        Student Number：1155261963
Hackathon：[AI Safety Hackathon](https://www.aisafetyhk.org/apart-hackathon-hk-hub)	Repository：[dasi0227/ftec5660-hackathon](https://gitee.com/dasi0227/ftec5660-hackathon)
Project：[Communication Limits, Oversight and Profit Pressure in LLM Auctions](https://apartresearch.com/sprints/projects/communication-limits-oversight-and-profit-pressure-in-llm-double-auctions-hux1)

## What is my project

My project explored whether restricting communication or introducing oversight could reduce coordination among AI sellers in a simulated market. Building on an existing double-auction simulator, I tested seven experimental conditions, each involving four AI buyers and four AI sellers trading over 30 rounds. These conditions varied sellers' access to chat, message length limits, Jev-based oversight, and pressure to maintain profits. I analyzed seller messages alongside subsequent quotes, transaction prices, and buyer and seller gains to understand how these factors affected market behavior.


## What worked well

**Refactored the multi-agent workflow and data structures:** I reworked the existing simulator so that buyers and sellers made decisions in parallel using the same market snapshot. I also improved how the simulator handled withdrawals, trades, memories, and strategy notes, and reorganized the data structures to better track market activity and oversight across rounds. These changes made it easier to follow how agents interacted and how their decisions evolved.

**Integrated Jev and designed comparative experiments:** I integrated Jev into the simulator to evaluate seller messages using context from the current and previous rounds. When the coordination score exceeded a threshold, sellers received a warning in the following round. I then designed seven conditions combining different communication limits, oversight settings, and profit pressure, while keeping other settings consistent. This allowed me to compare their effects under similar conditions.

**Improved code usability and analysis:** I cleaned up outdated files and unnecessary options, simplified the evaluation workflow, and standardized experiment configurations and outputs. Each run recorded its settings, model calls, market activity, and summary results, making it easier to trace individual decisions and connect seller messages to later trading behavior. I also added tests and improved response validation and retry handling after discovering an issue with missing fields.

## What worked poorly

**Relied on a single random seed:** I only ran each condition with one random seed, which meant I could not tell how consistent the results would be across repeated runs. Some of the differences I observed might have come from randomness in the agents' decisions rather than the experimental conditions themselves. This limited how confidently I could interpret the findings.

**Overlooked missing fields in the retry mechanism:** My initial validation did not catch every type of invalid model response. For example, a model might omit a required field when it intended to leave it empty, but the retry mechanism would not recognize this as an error.

**Used Jev without independent calibration:** I defined the scoring criteria and applied Jev directly without checking its judgments against independently labeled examples. As a result, some seller messages might have been misclassified, leading to warnings being issued or missed incorrectly.

**Did not sufficiently adapt the prompts to the model:** The original paper used GPT-4.1, while my experiments used a newer model, DeepSeek V4 Flash. Although I made some adjustments to the prompts, I did not systematically test which instructions worked best with this model. This may have affected how well the agents understood their roles and made trading decisions.

## What I would improve

**Expand the comparison groups:** I would explore more factors, especially the choice of model. Following the original paper, I would compare markets where all sellers use the same model with markets where sellers use different models, while keeping other settings unchanged. I would also extend the number of trading rounds to see how coordination develops over time and run each condition with multiple random seeds to check whether the findings are consistent.

**Introduce financial penalties:** Instead of relying only on written warnings, I would introduce simulated fines for sellers whose behavior meets a defined coordination criterion. By comparing warnings alone with fines of different amounts, I could test whether financial consequences are more effective at discouraging coordination. I would also look at how these penalties affect transaction prices, trading activity, and buyer outcomes.

**Apply a minimum profit requirement:** I would turn the pressure from a boss into an actual trading constraint rather than just an instruction in the prompt. For example, I could require sellers to earn at least $5 above their cost on every completed sale and reject quotes that fall below this limit. Comparing this with prompt-based pressure would help me understand whether explicit profit constraints have a different effect on seller coordination, transaction prices, and trading volume.

## What I have learned

This was my first hackathon, and it changed my perspective on how much I could accomplish in a short period of time. Throughout the event, I had to move between understanding the original code, rebuilding parts of it, conducting experiments, and turning the results into a report. The tight deadline pushed me to make quick decisions about what deserved further attention and when it was better to move forward. Seeing all these pieces come together into a complete project gave me greater confidence in my ability to tackle an unfamiliar problem and carry it through to a meaningful result. I also gained a better understanding of my own working habits: knowing where to focus my time and effort was just as important as the number of hours I put in.

In this project, I observed how agents' trading behavior changed under different prompts. As personal AI agents become increasingly popular, I believe we are moving toward a future where people may entrust agents with real digital currencies or payment capabilities to conduct transactions on their behalf. This made me think about the potential security risks involved. For example, an AI agent shopping on behalf of a user might encounter webpages or messages from sellers that contain instructions the user never authorized. If the agent mistakenly treats malicious content as legitimate commands, a prompt injection attack could manipulate it into making unauthorized purchases or transferring money to the wrong recipient.

Although my experiments involved simulated transactions, they helped me recognize the importance of controlling who and what can influence an agent's decisions. Once AI agents are capable of handling real money, preventing untrusted external content from overriding users' original instructions becomes essential to protecting their financial assets. More importantly, this hackathon made me realize that building capable AI agents is not enough; ensuring that they act safely and reliably is equally important. It also deepened my awareness of AI safety challenges and motivated me to explore the principles of secure agent design and the broader philosophy behind AI safety.
