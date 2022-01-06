# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.
import utils

eval_corpus_path = "birth_dev.tsv"
n = sum(1 for _ in open(eval_corpus_path,encoding="utf-8"))
predictions = ["London"]*n

total, correct = utils.evaluate_places(eval_corpus_path, predictions)
print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))