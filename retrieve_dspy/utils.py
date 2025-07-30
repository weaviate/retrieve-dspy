import dspy
from dspy import Example

def get_evaluator(
    testset: list[Example],
    metric: callable
):
    evaluator = dspy.Evaluate(
        devset=testset,
        metric=metric, 
        num_threads=1,
        display_progress=True,
        max_errors=1,
        provide_traceback=True
    )

    return evaluator