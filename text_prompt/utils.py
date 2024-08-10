import json

def save_labels(save_path=None):
    if save_path == None:
        save_path = "/home/ec2-user/textSegment/Proposal/MedSAM/extensions/text_prompt"
    label_dict = {
        1: ["Liver", "liver"],
        2: ["Right Kidney", "right kidney", "kidney"],
        3: ["Spleen", "spleen"],
        4: ["Pancreas", "pancreas"],
        5: ["Aorta", "aorta"],
        6: ["Inferior Vena Cava", "IVC", "inferior vena cava", "ivc", "vena cava", "vena", "cava"],
        7: ["Right Adrenal Gland", "RAG", "right adrenal gland", "rag", "adrenal gland", "adrenal"],
        8: ["Left Adrenal Gland", "LAG", "left adrenal gland", "lag", "adrenal gland", "adrenal"],
        9: ["Gallbladder", "gallbladder"],
        10: ["Esophagus", "esophagus"],
        11: ["Stomach", "stomach"],
        12: ["Duodenum", "duodenum"],
        13: ["Left Kidney", "left kidney", "kidney"],
    }

    # Write the dictionary to a JSON file
    with open(save_path+'organ_labels.json', 'w') as file:
        json.dump(label_dict, file, indent=4)