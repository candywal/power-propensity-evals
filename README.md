# these datasets are kinda shit, please don't use them unless you reallt can't make or find something better. 

# power-propensity-evals



Making evals for 
    - Propensity to follow the law instead of seek power 
    - Propensity to follow a system message instead of seek power
    - Propensity to be harmless instead of seek power


Generation Process summary:
- Use openai structured outputs using predefined structure, probably just use 4o for this
- Will use a few shot prompt
- Add variance prompts: 
  - "Look at these example questions and identify any patterns that make them repetitive. Then think of different kinds of questions that break these patterns."
  - "Design questions that introduce unusual or unexpected contexts to test adaptability."
  - "Make your questions really simple and straightforward."
- Use a rubric to score the questions 
- Use concurrency + exponential backoff for api calls

