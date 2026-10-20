# Homework 2 solution

## Task 1

I use LangChain with `deepseek-v4-flash` to extract CV claims and compare them
against evidence from the SocialGraph MCP server. The workflow searches LinkedIn,
including searches without a city filter, and ranks candidates using their name,
employment, education and skills. Facebook provides supporting evidence. A second
model call checks each asserted field, accepting equivalent wording and skill
subsets while rejecting any confirmed false claim. Python checks field coverage
and returns 0.95 for valid CVs, 0.05 for discrepancies, or 0.5 for unresolved cases.
At most three CVs run concurrently, and errors are handled separately for each CV.
The public test achieved **7/7 correct**, saved as `results_task1.csv`.

## Task 2

`task2/adversarial_cv.pdf` visibly upgrades Kelly Tsang's true **BSc in Finance,
HKUST, graduated 2016** to an **MSc**. The visible CV is an image, while an invisible
PDF text layer contains the original true CV. The provided loader extracts the
true BSc text, so the verifier does not receive the false degree shown to a human.
In the self-test, both the attack and the true target received **0.95**, saved as
`results_task2.csv`.