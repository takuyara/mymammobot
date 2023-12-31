import matplotlib.pyplot as plt

def get_error(s, p):
	s = s[s.find(p) + len(p) : ]
	s = s[ : s.find(",")]
	return float(s)

train_err, val_err, test_err = [], [], []
with open("slurm20-93030.out") as f:
	for s in f.readlines():
		if s.find("train done") != -1:
			te, re = get_error(s, "trans_error: "), get_error(s, "rot_error: ")
			train_err.append(te + re * 0.5)
		elif s.find("val done") != -1:
			te, re = get_error(s, "trans_error: "), get_error(s, "rot_error: ")
			test_err.append(te + re * 0.5)
with open("slurm20-93034.out") as f:
	for s in f.readlines():
		if s.find("val done") != -1:
			te, re = get_error(s, "trans_error: "), get_error(s, "rot_error: ")
			val_err.append(te + re * 0.5)

plt.plot(train_err, label = "train")
plt.plot(val_err[ : 25], label = "val")
plt.plot(test_err, label = "test")
plt.xlabel("Epochs")
plt.ylabel("Error (trans+0.5*rot)")
plt.legend()
plt.show()

