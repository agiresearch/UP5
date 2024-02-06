"""
Pretraining Tasks -- 5 Prompt Families (1, 2, 3, 4, 5)
Zeroshot Tasks -- 1 Prompt Family (Z)
"""

all_tasks = {}


# =====================================================
# Task Subgroup 2 -- Sequential -- 13 Prompts
# =====================================================

task_subgroup_2 = []

template = {}

"""
Input template:
Given the following purchase history of user {{user_id}}:
{{history item list of {{item_id}}}}
predict next possible item to be purchased by the user?


Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
"""

template[
    "source"
] = "User_{} has purchased {}, predict next possible insurance to be bought by the user ?"
template["target"] = "{}"
template["task"] = "sequential"
template["id"] = "2-1"

task_subgroup_2.append(template)


template = {}
"""
Input template:
I find the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}
I wonder which is the next item to recommend to the user. Can you help me decide?


Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
"""
template[
    "source"
] = "I find the purchase list of user_{} : \n {} \n I wonder what other insurance does the user need. Can you help me decide ?"
template["target"] = "{}"
template["task"] = "sequential"
template["id"] = "2-2"

task_subgroup_2.append(template)


template = {}
"""
Input template:
Here is the purchase history list of user {{user_id}}:
{{history item list of {{item_id}}}}
try to recommend next item to the user

Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
"""
template[
    "source"
] = "Here is the purchase list of user_{} : \n {} \n try to recommend another insurance to the user."
template["target"] = "{}"
template["task"] = "sequential"
template["id"] = "2-3"

task_subgroup_2.append(template)


template = {}
"""
Input template:
According to the purchase history of {{user_desc}}:
{{history item list of {{item_id}}}}
Can you recommend the next possible item to the user?

Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
"""
template[
    "source"
] = "According to what insurances user_{} has purchased: \n {} \n Can you recommend another insurance to the user ?"
template["target"] = "{}"
template["task"] = "sequential"
template["source_argc"] = 2
template["source_argv"] = ["user_desc", "purchase_history"]
template["target_argc"] = 1
template["target_argv"] = ["item_id"]
template["id"] = "2-4"

task_subgroup_2.append(template)


template = {}
"""
Input template:
Based on the purchase history of {{user_desc}}:
{{history item list of {{item_id}}}}
Can you decide the next item likely to be purchased by the user?


Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
"""
template[
    "source"
] = "The user_{} has bought insurances: \n {} \n What else do you think is necessary for the user ?"
template["target"] = "{}"
template["task"] = "sequential"
template["id"] = "2-5"

task_subgroup_2.append(template)


template = {}
"""
Input template:
Here is the purchase history of {{user_desc}}:
{{history item list of {{item_id}}}}
What to recommend next for the user?

Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
"""
template[
    "source"
] = "Here is the insurance purchase history of user_{} : \n {} \n What to recommend next for the user ?"
template["target"] = "{}"
template["task"] = "sequential"
template["id"] = "2-6"

task_subgroup_2.append(template)


# Extractive QA
template = {}
"""
Input template:
Here is the purchase history of user {{user_id}}:
{{history item list of {{item_id}}}}
Select the next possible item likely to be purchased by the user from the following candidates:
{{candidate {{item_id}}}}


Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
"""
template[
    "source"
] = "Here is the insurance history of user_{} : \n {} \n Select the most useful insurance to the user from the following candidates : \n {}"
template["target"] = "{}"
template["task"] = "sequential"
template["id"] = "2-7"

task_subgroup_2.append(template)


template = {}
"""
Input template:
Given the following purchase history of {{user_desc}}:
{{history item list of {{item_id}}}}
What to recommend next for the user? Select one from the following items:
{{candidate {{item_id}}}}

Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
"""
template[
    "source"
] = "Given the following insurance bought by user_{} : \n {} \n What is most likely to be bought by the user? Select one from the following insurances : \n {}"
template["target"] = "{}"
template["task"] = "sequential"
template["source_argc"] = 3
template["source_argv"] = ["user_desc", "purchase_history", "candidates"]
template["target_argc"] = 1
template["target_argv"] = ["item_id"]
template["id"] = "2-8"

task_subgroup_2.append(template)


template = {}
"""
Input template:
Based on the purchase history of user {{user_id}}:
{{history item list of {{item_id}}}}
Choose the next possible purchased item from the following candidates:
{{candidate {{item_id}}}}


Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
"""
template[
    "source"
] = "Based on purchased insurances of user_{} : \n {} \n Choose the most helpful insurance from the following candidates : \n {}"
template["target"] = "{}"
template["task"] = "sequential"
template["source_argc"] = 3
template["source_argv"] = ["user_id", "purchase_history", "candidates"]
template["target_argc"] = 1
template["target_argv"] = ["item_id"]
template["id"] = "2-9"

task_subgroup_2.append(template)

template = {}
"""
Input template:
I find the purchase history list of {{user_desc}}:
{{history item list of {{item_id}}}}
I wonder which is the next item to recommend to the user. Try to select one from the following candidates:
{{candidate {{item_id}}}}

Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
"""
template[
    "source"
] = "I find the bought insurance list of user_{} : \n {} \n I wonder which is the next insurance to recommend to the user . Try to select one from the following candidates : \n {}"
template["target"] = "{}"
template["task"] = "sequential"
template["source_argc"] = 3
template["source_argv"] = ["user_desc", "purchase_history", "candidates"]
template["target_argc"] = 1
template["target_argv"] = ["item_id"]
template["id"] = "2-10"

task_subgroup_2.append(template)


# Pairwise Prediction
template = {}
"""
Input template:
User {{user_id}} has the following purchase history:
{{history item list of {{item_id}}}}
Does the user likely to buy {{item [item_id]}} next?

Target template:
{{answer_choices[label]}} (yes/no)

Metrics:
Accuracy
"""
template[
    "source"
] = "User_{} has bought the following insurances : \n {} \n does the user likely to buy {} as well ?"
template["target"] = "{}"
template["task"] = "sequential"
template["source_argc"] = 3
template["source_argv"] = ["user_id", "purchase_history", "item_id"]
template["target_argc"] = 1
template["target_argv"] = ["yes_no"]
template["id"] = "2-11"

task_subgroup_2.append(template)


template = {}
"""
Input template:
According to {{user_desc}}'s purchase history list:
{{history item list of {{item_id}}}}
Predict whether the user will purchase {{item [item_id]}} next?

Target template:
{{answer_choices[label]}} (yes/no)

Metrics:
Accuracy
"""
template[
    "source"
] = "According to user_{} 's bought insurance list : \n {} \n Predict whether the user will buy or has already bought {} ."
template["target"] = "{}"
template["task"] = "sequential"
template["source_argc"] = 3
template["source_argv"] = ["user_desc", "purchase_history", "item_id"]
template["target_argc"] = 1
template["target_argv"] = ["yes_no"]
template["id"] = "2-12"

task_subgroup_2.append(template)

template = {}
"""
Input template:
According to {{user_desc}}'s purchase history list:
{{history item list of {{item_id}}}}
Predict whether the user will purchase {{item [item_id]}} next?

Target template:
{{answer_choices[label]}} (yes/no)

Metrics:
Accuracy
"""
template[
    "source"
] = "According to user_{} 's bought insurance list : \n {} \n Do you think the user also needs {} ?"
template["target"] = "{}"
template["task"] = "sequential"
template["source_argc"] = 3
template["source_argv"] = ["user_desc", "purchase_history", "item_id"]
template["target_argc"] = 1
template["target_argv"] = ["yes_no"]
template["id"] = "2-13"

task_subgroup_2.append(template)

all_tasks["sequential"] = task_subgroup_2


# =====================================================
# Task Subgroup 5 -- Traditional -- 8 Prompts
# =====================================================

task_subgroup_5 = []

template = {}

"""
Input template:
Will user {{user_id}} likely to interact with item {{item_id}}?


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
"""

template["source"] = "Will user_{} likely to purchase insurance_{} ?"
template["target"] = "{}"
template["task"] = "traditional"
template["source_argc"] = 2
template["source_argv"] = ["user_id", "item_id"]
template["target_argc"] = 1
template["target_argv"] = ["yes_no"]
template["id"] = "5-1"

task_subgroup_5.append(template)


template = {}

"""
Input template:
Shall we recommend item {{item_id}} to {{user_desc}}?


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
"""

template["source"] = "Shall we recommend insurance_{} to user_{} ?"
template["target"] = "{}"
template["task"] = "traditional"
template["source_argc"] = 2
template["source_argv"] = ["item_id", "user_desc"]
template["target_argc"] = 1
template["target_argv"] = ["yes_no"]
template["id"] = "5-2"

task_subgroup_5.append(template)


template = {}

"""
Input template:
For {{user_desc}}, do you think it is good to recommend {{item_title}}?


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
"""

template["source"] = "For user_{}, do you think it is good to recommend insurance_{} ?"
template["target"] = "{}"
template["task"] = "traditional"
template["source_argc"] = 2
template["source_argv"] = ["user_desc", "item_title"]
template["target_argc"] = 1
template["target_argv"] = ["yes_no"]
template["id"] = "5-3"

task_subgroup_5.append(template)


template = {}

"""
Input template:
I would like to recommend some items for user {{user_id}}. Is the following item a good choice?
{{item_title}}


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
"""

template[
    "source"
] = "I would like to recommend some insurance for user_{} . Is the following insurance a good choice ? \n insurance_{}"
template["target"] = "{}"
template["task"] = "traditional"
template["source_argc"] = 2
template["source_argv"] = ["user_id", "item_title"]
template["target_argc"] = 1
template["target_argv"] = ["yes_no"]
template["id"] = "5-4"

task_subgroup_5.append(template)


template = {}

"""
Input template:
I would like to recommend some items for user {{user_id}}. Is the following item a good choice?
{{item_title}}


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
"""

template["source"] = "Do you think user_{} will like to buy insurance_{} ?"
template["target"] = "{}"
template["task"] = "traditional"
template["source_argc"] = 2
template["source_argv"] = ["user_id", "item_title"]
template["target_argc"] = 1
template["target_argv"] = ["yes_no"]
template["id"] = "5-5"

task_subgroup_5.append(template)


template = {}

"""
Input template:
I would like to recommend some items for user {{user_id}}. Is the following item a good choice?
{{item_title}}


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
"""

template[
    "source"
] = "I would like to find what insurance user_{} needs. What about {} ?"
template["target"] = "{}"
template["task"] = "traditional"
template["source_argc"] = 2
template["source_argv"] = ["user_id", "item_title"]
template["target_argc"] = 1
template["target_argv"] = ["yes_no"]
template["id"] = "5-6"

task_subgroup_5.append(template)


template = {}

"""
Input template:
I would like to recommend some items for user {{user_id}}. Is the following item a good choice?
{{item_title}}


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
"""

template[
    "source"
] = "I would like to recommend some insurances for user_{} . Is the following one a good choice ? \n {}"
template["target"] = "{}"
template["task"] = "traditional"
template["source_argc"] = 2
template["source_argv"] = ["user_id", "item_title"]
template["target_argc"] = 1
template["target_argv"] = ["yes_no"]
template["id"] = "5-7"

task_subgroup_5.append(template)


template = {}

"""
Input template:
Which item of the following to recommend for {{user_desc}}?
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
"""

template["source"] = "Which insurance of the following to recommend for user_{} ? \n {}"
template["target"] = "{}"
template["task"] = "traditional"
template["source_argc"] = 2
template["source_argv"] = ["user_desc", "candidates"]
template["target_argc"] = 1
template["target_argv"] = ["groundtruth_item_ids"]
template["id"] = "5-8"

task_subgroup_5.append(template)

template = {}

"""
Input template:
Choose the best item from the candidates to recommend for {{user_desc}}?
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
"""

template[
    "source"
] = "Choose the best insurance from the candidates to recommend for user_{} ? \n {}"
template["target"] = "{}"
template["task"] = "traditional"
template["source_argc"] = 2
template["source_argv"] = ["user_desc", "candidates"]
template["target_argc"] = 1
template["target_argv"] = ["groundtruth_item_ids"]
template["id"] = "5-9"

task_subgroup_5.append(template)


template = {}

"""
Input template:
Pick the most suitable item from the following list and recommend to user {{user_id}}:
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
"""

template[
    "source"
] = "Pick the most useful insurance from the following list and recommend to user_{} : \n {}"
template["target"] = "{}"
template["task"] = "traditional"
template["source_argc"] = 2
template["source_argv"] = ["user_id", "candidates"]
template["target_argc"] = 1
template["target_argv"] = ["groundtruth_item_ids"]
template["id"] = "5-10"

task_subgroup_5.append(template)


template = {}

"""
Input template:
We want to make recommendation for user {{user_id}}. Select the best item from these candidates:
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
"""

template[
    "source"
] = "We want to make recommendation for user_{} .  Select the best insurance from these candidates : \n {}"
template["target"] = "{}"
template["task"] = "traditional"
template["source_argc"] = 2
template["source_argv"] = ["user_id", "candidates"]
template["target_argc"] = 1
template["target_argv"] = ["groundtruth_item_ids"]
template["id"] = "5-11"

task_subgroup_5.append(template)


all_tasks["traditional"] = task_subgroup_5


# ========================================================
# Cold-Start/Zero-Shot Task Subgroup - 7 Prompts
# ========================================================

"""
Zero-Shot Inference Tasks
"""

zero_short_tasks = {}

template = {}

"""
Input template:
Given the facts about the new product, do you think user {{user_id}} will like or dislike it?
title: {{item_title}}
brand: {{brand}}
price: {{price}}


Target template:
{{answer_choices[label]}} (like/dislike) – like (4,5) / dislike (1,2,3)


Metrics:
Accuracy
"""

template[
    "source"
] = "Given the facts about the new product , do you think user_{} will like or dislike it ? \n title : {} \n brand : {} \n price : {}"
template["target"] = "{}"
template["task"] = "zeroshot"
template["source_argc"] = 4
template["source_argv"] = ["user_id", "item_title", "brand", "price"]
template["target_argc"] = 1
template["target_argv"] = ["like_dislike"]
template["id"] = "Z-1"

zero_short_tasks["Z-1"] = template


template = {}

"""
Input template:
Here are the details about a new product:
title: {{item_title}}
brand: {{brand}}
price: {{price}}
What star will {{user_desc}} probably rate the product?
-1
-2
-3
-4
-5

Target template:
{{answer_choices[star_rating-1]}}


Metrics:
Accuracy
"""

template[
    "source"
] = "Here are the details about a new product : \n title : {} \n brand : {} \n price : {} \n What star will {} probably rate the product ? \n -1 \n -2 \n -3 \n -4 \n -5"
template["target"] = "{}"
template["task"] = "zeroshot"
template["source_argc"] = 4
template["source_argv"] = ["item_title", "brand", "price", "user_desc"]
template["target_argc"] = 1
template["target_argv"] = ["star_rating"]
template["id"] = "Z-2"

zero_short_tasks["Z-2"] = template


template = {}

"""
Input template:
Predict user {{user_id}}'s preference about the new product (1 being lowest and 5 being highest):
title: {{item_title}}
price: {{price}}
brand: {{brand}}


Target template:
{{answer_choices[star_rating-1]}}


Metrics:
Accuracy
"""

template[
    "source"
] = "Predict user_{} 's preference about the new product ( 1 being lowest and 5 being highest ) : \n title : {} \n price : {} \n brand : {}"
template["target"] = "{}"
template["task"] = "zeroshot"
template["source_argc"] = 4
template["source_argv"] = ["user_id", "item_title", "price", "brand"]
template["target_argc"] = 1
template["target_argv"] = ["star_rating"]
template["id"] = "Z-3"

zero_short_tasks["Z-3"] = template


template = {}

"""
Input template:
Will {{user_desc}} like or dislike the following product?
title: {{item_title}}
price: {{price}}
brand: {{brand}}

Target template:
{{answer_choices[label]}} (like/dislike) – like (4,5) / dislike (1,2,3)


Metrics:
Accuracy
"""

template[
    "source"
] = "Will {} like or dislike the following product ? \n title : {} \n price : {} \n brand : {}"
template["target"] = "{}"
template["task"] = "zeroshot"
template["source_argc"] = 4
template["source_argv"] = ["user_desc", "item_title", "price", "brand"]
template["target_argc"] = 1
template["target_argv"] = ["like_dislike"]
template["id"] = "Z-4"

zero_short_tasks["Z-4"] = template


template = {}

"""
Input template:
Generate a possible explanation for {{user_desc}}'s preference about the following product:
title: {{item_title}}
brand: {{brand}}
price: {{price}}

Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
"""

template[
    "source"
] = "Generate a possible explanation for {} 's preference about the following product : \n title : {} \n brand : {} \n price : {}"
template["target"] = "{}"
template["task"] = "zeroshot"
template["source_argc"] = 4
template["source_argv"] = ["user_desc", "item_title", "brand", "price"]
template["target_argc"] = 1
template["target_argv"] = ["explanation"]
template["id"] = "Z-5"

zero_short_tasks["Z-5"] = template


template = {}

"""
Input template:
Based on the word {{feature}}, help user {{user_id}} write a {{star_rating}}-star explanation for this new product:
title: {{item_title}}
price: {{price}}
brand: {{brand}}

Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
"""

template[
    "source"
] = "Based on the word {} , help user_{} write a {}-star explanation for this new product : \n title : {} \n price : {} \n brand : {}"
template["target"] = "{}"
template["task"] = "zeroshot"
template["source_argc"] = 6
template["source_argv"] = [
    "feature",
    "user_id",
    "star_rating",
    "item_title",
    "price",
    "brand",
]
template["target_argc"] = 1
template["target_argv"] = ["explanation"]
template["id"] = "Z-6"

zero_short_tasks["Z-6"] = template


template = {}

"""
Input template:
For the new product {{item_title}}, we would like to know whether {{user_desc}} will love it. If you think the user will love it, please help explain why.


Target template:
{{explanation}}


Metrics:
BLUE, ROUGE
"""

template[
    "source"
] = "For the new product {} , we would like to know whether {} will love it . If you think the user will love it , please help explain why ."
template["target"] = "{}"
template["task"] = "zeroshot"
template["source_argc"] = 2
template["source_argv"] = ["item_title", "user_desc"]
template["target_argc"] = 1
template["target_argv"] = ["explanation"]
template["id"] = "Z-7"

zero_short_tasks["Z-7"] = template
