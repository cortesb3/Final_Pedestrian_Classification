"""
    Script to check if there is any misalignment on GT mask and person within a folder
"""
from glob import glob

mydir ="final_filtered_pedestrian_pool"

mask_list = glob(mydir + "/*_mask.png")
person_list = glob(mydir + "/*_person.png")

final_mask =[]
final_person = []

for mask in mask_list:
    final_mask.append(mask[:-8])
for person in person_list:
    final_person.append(person[:-10])

print(f'size of filtered pedestrian file: {len(final_mask+final_person)}')
print(f'Extra person.png: {list(set(final_person) - set(final_mask))}')
print(f'Extra mask.png: {list(set(final_mask) - set(final_person))}')