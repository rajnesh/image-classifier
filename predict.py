#usage example: #python predict.py flowers/test/1/image_06743.jpg vgg_cp/checkpoint.pth -gpu --top_k 2

from get_input_args_p import get_input_args
from pred_utility import load_and_rebuild, process_image, predict
import json

in_arg = get_input_args()
with open(in_arg.category_names, 'r') as f:
    cat_to_name = json.load(f)

model, optimizer = load_and_rebuild(in_arg.checkpoint, in_arg.gpu)
top_probs, top_classes, actual_class = predict(in_arg.image_path, in_arg.gpu, model, in_arg.top_k)

top_class_names = []
for c in top_classes:
    top_class_names.append(cat_to_name[c])

#print("Actual Class: " + cat_to_name[actual_class]) #uncomment to print actual class name if testing

if in_arg.top_k == 1:
    print("Predicted Class: {}".format(top_class_names[0]))
    print("Class Probability: %d %%" % (top_probs[0]*100))
else:
    print("Top {} classes and their probabilities are as below:".format(in_arg.top_k))
    for ii in range(0, in_arg.top_k):
        if (top_probs[ii]*100) < 1:
            print("{}.  {}  less than 1%".format(ii+1, top_class_names[ii]))
        else:
            print("{}.  {}  {}%".format(ii+1, top_class_names[ii], (top_probs[ii]*100).astype(int)))