int_to_label = lambda x: "against" if x==0 else "for" if x==1 else f"error: label is neither 0 nor 1 is {x}"
label_to_int = lambda x: 0 if x == "against" else 1 if x=="for" else -1