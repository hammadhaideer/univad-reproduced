import json
import os


class MVTecLOCOSolver:
    CLSNAMES = [
        'breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors',
    ]

    def __init__(self, root='./data/mvtec_loco_caption'):
        self.root = root
        self.meta_path = os.path.join(root, 'meta.json')

    def run(self):
        info = dict(train={}, test={})
        for cls_name in self.CLSNAMES:
            cls_dir = os.path.join(self.root, cls_name)
            for phase in ['train', 'test']:
                cls_info = []
                species = os.listdir(os.path.join(cls_dir, phase))
                for specie in species:
                    is_abnormal = specie not in ['good']
                    img_names = os.listdir(os.path.join(cls_dir, phase, specie))
                    mask_names = (
                        os.listdir(
                            os.path.join(
                                cls_dir, 'ground_truth_merge_mask', f'{specie}_merge_mask'
                            )
                        )
                        if is_abnormal
                        else None
                    )
                    img_names.sort()
                    if mask_names is not None:
                        mask_names.sort()
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=os.path.join(cls_name, phase, specie, img_name),
                            mask_path=(
                                os.path.join(
                                    cls_name, 'ground_truth_merge_mask',
                                    f'{specie}_merge_mask', mask_names[idx],
                                )
                                if is_abnormal
                                else ''
                            ),
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")


if __name__ == '__main__':
    runner = MVTecLOCOSolver(root='./data/mvtec_loco_caption')
    runner.run()