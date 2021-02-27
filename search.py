import os
from ray import tune
from functools import partial

from transformers import DistilBertTokenizerFast

from args import get_train_test_args
from train import do_train

def main():
    if not os.path.exists("tune_results"):
        os.makedirs("tune_results")

    args = get_train_test_args()
    args["tune"] = True

    args["lr"] = tune.loguniform(1e-4, 1e-1)
    args["batch_size"] = tune.choice(args["tune_batch_sizes"])
    args["seed"] = tune.randint(1, 100)
    args["adam_weight_decay"] = tune.loguniform(1e-4, 1e-1)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    result = tune.run(
        partial(do_train, tokenizer=tokenizer),
        config=args,
        local_dir="tune_results",
        num_samples=args["num_tune_samples"],
        resources_per_trial={"cpu": args["num_cpu_per_test"], "gpu": args["num_gpu_per_test"]},
    )
    
    # trial results automatically get logged by tune
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {0}".format(best_trial.config))
    print("Best trial final loss: {0}".format(best_trial.last_result["loss"]))
    print("Best trial final F1: {0}".format(best_trial.last_result["F1"]))
    print("Best trial final EM: {0}".format(best_trial.last_result["EM"]))

if __name__ == "__main__":
    main()