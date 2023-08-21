templates_list = {
    "0SAU": "'[S_CONTEXT]' In previous sentences, does '[MENTION2]' refer to '[MENTION1]'? Yes or no?",
    "1SAU": "'[S_CONTEXT]' Here, by '[MENTION2]' they mean '[MENTION1]'? Yes or no?",
    "2SAU": "'[S_CONTEXT]' Here, does '[MENTION2]' stand for '[MENTION1]'? Yes or no? ",
    "3SAU": "'[S_CONTEXT]' In the passage above, can '[MENTION2]' be replaced by '[MENTION1]'? Yes or no?",
    "4SAU": "'[S_CONTEXT]' I think '[MENTION2]' means '[MENTION1]'. Yes or no?",
    "5SAU": "Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred. Now, please tell me whether the mention \"[MENTION2]\" refer to the mention \"[MENTION1]\" in the following sentence: \"[S_CONTEXT]\" Yes or No?",
    "6SAU": "Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred. Now, please tell me whether the mention \"[MENTION2]\" is co-referred with the mention \"[MENTION1]\" in the following sentence: \"[S_CONTEXT]\" Yes or No?",
    "7SAU": """Please determine whether two mentions in a given text are co-referred. Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred. 
Use the following format:
Text: <the given text>
Mention1: <one mention in the given text>
Mention2: <another mention in the given text>
Answer: <Yes or No>

Text: \"\"\"[S_CONTEXT] \"\"\"
Mention1: [MENTION1]
Mention2: [MENTION2]
Answer:""",
    "8SAU": """Please determine whether two mentions in a given text are co-referred. Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred. 
Use the following format:
Text: <the given text>
Mention1: <one mention in the given text>
Mention2: <another mention in the given text>
Answer: <Yes or No>

Text:\"\"\"[S_CONTEXT] \"\"\"
Mention1: [MENTION1]
Mention2: [MENTION2]
Answer:""",
    "9SAU": """Please determine whether two mentions in a given text are co-referred. Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred. 
Use the following format:
Text: <the given text>
Mention1: <one mention in the given text>
Mention2: <another mention in the given text>
Answer: <Yes or No>

Text:\"\"\"[S_CONTEXT] \"\"\"
Mention1:" [MENTION1] "
Mention2:" [MENTION2] "
Answer:""",
    "10SAU": """Please determine whether two mentions in a given text are co-referred. Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred. 
Use the following format:
Text: <the given text>
Mention1: <one mention in the given text>
Mention2: <another mention in the given text>
Answer: <Yes or No>

Text: < [S_CONTEXT] >
Mention1: < [MENTION1] >
Mention2: < [MENTION2] >
Answer:""",
    "11SAU": """Please determine whether two mentions in a given text are co-referred. Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred. 
Use the following format:
Text: < the given text >
Mention1: < one mention in the given text >
Mention2: < another mention in the given text >
Answer: < Yes or No >

Text: < [S_CONTEXT] >
Mention1: < [MENTION1] >
Mention2: < [MENTION2] >
Answer:""",
    "12SAU": """Please determine whether two mentions in a given text are co-referred. Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred.
Use the following format:
Text: < the given text >
Mention1: < one mention in the given text >
Mention2: < another mention in the given text >
Answer: < Yes or No >

Text: < [S_CONTEXT] >
Mention1: < [MENTION1] >
Mention2: < [MENTION2] >
Answer:""",  # 相比于prompt11， 第一行删了最后的空格
    "13SAU": """Please determine whether two mentions in a given text are co-referred. Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred.
Use the following format:
Text: < the given text >
Mention1: < one mention in the given text >
Mention2: < another mention in the given text >
Answer: < Yes or No, why >

Text: < [S_CONTEXT] >
Mention1: < [MENTION1] >
Mention2: < [MENTION2] >
Answer:""",
    "15DAM": """You will be given one or two messages (if there are two, they are separated by $$). In the given messages, two mentions are marked by <mention></mention> label. Please determine whether two mentions are co-referred. Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of anther mention, the two mentions don't co-referred.
Use the following format:
Text: < the given messages with two mentions marked >
Answer: < Yes or No, why >

Text: < [D_CONTEXT_marked] >
Answer:""",
    "16DAM": """You will be given one or two messages (if there are two, they are separated by $$). In the given messages, two mentions are marked by <mention></mention> label. Please determine whether two mentions are co-referred. Two mentions are co-referred only when they are refer to the identical entity or event. Note that if a mention is a side, a part, a view or a aspect of anther mention, the two mentions don't co-referred. Note that entity mention and event mention can not be coreferred.
Use the following format:
Text: < the given messages with two mentions marked >
Answer: < Yes or No, why >

Text: < [D_CONTEXT_marked] >
Answer:"""
}