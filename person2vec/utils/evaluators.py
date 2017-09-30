from numpy import reshape

from person2vec.test_embeddings import test_tasks

def get_chosen_categories(preds, category_list):
    chosen_categories = []
    for i in range(0, len(preds[0])):
        if preds[0][i] > 0:
            chosen_categories.append(category_list[i])
    return chosen_categories


def get_yelp_category_results(model, test_model, category_list, db_handler,
                            data_gen, embed_size, threshold=0.15):
    embeds = test_tasks.get_embed_weights_from_model(model)
    ids_embeds = test_tasks.reassociate_embeds_with_ids(embeds, data_gen=data_gen)
    ids_embeds.sort_index(inplace=True)
    chosen_and_correct_categories = []
    for row in ids_embeds.iterrows():#range(0, len(ids_embeds)):
        preds = test_model.predict(reshape(row[1], (1, embed_size)))
        preds[preds>=threshold] = 1
        preds[preds<threshold] = 0
        chosen_categories = get_chosen_categories(preds, category_list)
        correct_categories = db_handler.get_entity({'_id':row[0]})['categories']
        chosen_and_correct_categories.append((chosen_categories, correct_categories))
    return chosen_and_correct_categories


# takes a results in form of array of tuples, each tuple having 2 arrays
# first array in tuple is model output, second array in tuple is correct labels
def get_precision_recall(results):
    true_pos = 0.
    false_pos = 0.
    total_labels = 0.
    for result in results:
        outputs = result[0]
        labels = result[1]
        total_labels += len(labels)
        for output in outputs:
            if output in labels:
                true_pos += 1
            else:
                false_pos += 1

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / total_labels

    return precision, recall


def evaluate_yelp_category_results(model, test_model, category_list,
                                db_handler, data_gen, embed_size, threshold=0.15):
    results = get_yelp_category_results(model, test_model, category_list,
                                            db_handler, data_gen, embed_size, threshold)
    precision, recall = get_precision_recall(results[2300:])

    return {'precision':precision, 'recall':recall}
