templates = {
    1: "'[TEXT]' In previous sentences, does '[MENTION2]' refer to '[MENTION1]'? Yes or no?",

    2: "'[TEXT]' Here, by '[MENTION2]' they mean '[MENTION1]'? Yes or no?",

    3: "'[TEXT]' Here, does '[MENTION2]' stand for '[MENTION1]'? Yes or no? ",

    4: "'[TEXT]' In the passage above, can '[MENTION2]' be replaced by '[MENTION1]'? Yes or no?",

    5: "'[TEXT]' I think '[MENTION2]' means '[MENTION1]'. Yes or no?",

    6: "Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred. Now, please tell me whether the mention \"[MENTION2]\" refer to the mention \"[MENTION1]\" in the following sentence: \"[TEXT]\" Yes or No?",

    7: "Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred. Now, please tell me whether the mention \"[MENTION2]\" is co-referred with the mention \"[MENTION1]\" in the following sentence: \"[TEXT]\" Yes or No?",

    8: """Please determine whether two mentions in a given text are co-referred. Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred. 
Use the following format:
Text: <the given text>
Mention1: <one mention in the given text>
Mention2: <another mention in the given text>
Answer: <Yes or No>

Text: \"\"\"[TEXT] \"\"\"
Mention1: [MENTION1]
Mention2: [MENTION2]
Answer:""",

    9: """Please determine whether two mentions in a given text are co-referred. Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred. 
Use the following format:
Text: <the given text>
Mention1: <one mention in the given text>
Mention2: <another mention in the given text>
Answer: <Yes or No>

Text:\"\"\"[TEXT] \"\"\"
Mention1: [MENTION1]
Mention2: [MENTION2]
Answer:""",

    10: """Please determine whether two mentions in a given text are co-referred. Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred. 
Use the following format:
Text: <the given text>
Mention1: <one mention in the given text>
Mention2: <another mention in the given text>
Answer: <Yes or No>

Text:\"\"\"[TEXT] \"\"\"
Mention1:" [MENTION1] "
Mention2:" [MENTION2] "
Answer:""",

    11: """Please determine whether two mentions in a given text are co-referred. Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred. 
Use the following format:
Text: <the given text>
Mention1: <one mention in the given text>
Mention2: <another mention in the given text>
Answer: <Yes or No>

Text: < [TEXT] >
Mention1: < [MENTION1] >
Mention2: < [MENTION2] >
Answer:""",

    12: """Please determine whether two mentions in a given text are co-referred. Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred. 
Use the following format:
Text: < the given text >
Mention1: < one mention in the given text >
Mention2: < another mention in the given text >
Answer: < Yes or No >

Text: < [TEXT] >
Mention1: < [MENTION1] >
Mention2: < [MENTION2] >
Answer:""",

    13: """Please determine whether two mentions in a given text are co-referred. Two mention are co-referred only when they are refer to the identical thing. Note that if a mention is a side, a part, a view or a aspect of aonther mention, the two mentions don't co-referred.
Use the following format:
Text: < the given text >
Mention1: < one mention in the given text >
Mention2: < another mention in the given text >
Answer: < Yes or No >

Text: < [TEXT] >
Mention1: < [MENTION1] >
Mention2: < [MENTION2] >
Answer:"""  # 相比于prompt11， 第一行删了最后的空格
}
