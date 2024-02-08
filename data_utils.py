get_image_url = lambda x:x["inputs"]["image"]["url"]
get_image_dim = lambda x:{"width": x["inputs"]["image"]["width"], "height": x["inputs"]["image"]["height"]}
get_bboxes = lambda x:x["inputs"]["bboxes"]
get_clue = lambda x:x["inputs"]["clue"]
get_inference = lambda x:x["targets"]["inference"]
get_instance_id = lambda x:x["instance_id"]