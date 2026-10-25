import numpy as np


data = np.load("asl_landmarks_non.npz", allow_pickle=True)

# print(type(data))
# print(type(data["classes"]))
# print(type(data["X"]))
# print(type(data["y"]))
# print(data.files)


image = data["X"]
label = data["y"]
print("--------------------------------------------------------------")

# print(image[1], label[1])
# print(image[177],label[177])

ans = {}
summer = 0 
for i in range(29):
    num = 0
    for n,im in enumerate(image):
        if label[n] == i:
            num+=1
            summer+=1
    ans[i] = (num/3000)*100


sorted_pairs = sorted(
    ans.items(),
    key=lambda item: item[1]
)

sorted= dict(sorted_pairs)

for conv in sorted:
    print(f"{conv+1} : {sorted[conv]:.2f}")

# print(ans)
print(summer)