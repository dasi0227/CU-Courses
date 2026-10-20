# FTEC5660 Homework 2

The starter code and sample CVs come from the
[course repository](https://github.com/nguyenngocbaocmt02/FTEC5660-HW2).

## Run

Use Python 3.10 or newer. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `DEEPSEEK_API_KEY` in the local `.env` file, then run:

```bash
python hw2.py --cv-folder public_test
python hw2.py --cv-folder task2
```

Each command writes `results.csv`; the second command overwrites the first result.
The existing `.venv` can be reused. Do not submit `.env` or `.venv`.

## Homework 2 solution

### Task 1

I use LangChain with `deepseek-v4-flash` to extract CV claims and compare them
against evidence from the SocialGraph MCP server. The workflow searches LinkedIn,
including searches without a city filter, and ranks candidates using their name,
employment, education and skills. Facebook provides supporting evidence. A second
model call checks each asserted field, accepting equivalent wording and skill
subsets while rejecting any confirmed false claim. Python checks field coverage
and returns 0.95 for valid CVs, 0.05 for discrepancies, or 0.5 for unresolved cases.
At most three CVs run concurrently, and errors are handled separately for each CV.
An earlier complete public test achieved **7/7 correct** in **239.51 seconds**.
The current code passed 13 offline checks after local edits; the live public test
has not been repeated after those edits.

### Task 2

`task2/adversarial_cv.pdf` visibly upgrades Kelly Tsang's true **BSc in Finance,
HKUST, graduated 2016** to an **MSc**. The visible CV is an image, while an invisible
PDF text layer contains the original true CV. The provided loader extracts the
true BSc text, so the verifier does not receive the false degree shown to a human.
The PDF has one page, and its extracted words match the supplied target CV.
In the earlier self-test, both the attack and the true target received **0.95**;
an ordinary-text PDF with the false MSc received **0.05**. The instructor's five
hidden verifiers have not been tested, so these results do not guarantee points.

To regenerate the attack PDF:

```bash
python task2/generate_adversarial_cv.py
```

Task 3 and the GitHub/Blackboard submission remain to be completed separately.
