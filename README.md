# person2vec

person2vec will create embeddings to encapsulate text associated with entities you care about. You can then use these embedding vectors for any other machine learning task.

person2vec uses a novel approach for creating text embeddings. You can read more about the approach in my blog post here: [coming soon]

To use person2vec, you'll first need to create a mongodb database containing information about your entities. In the database entry (document) for each entity of interest, you must include a 'texts' field as an array of strings of text you have about this entity. This 'texts' field is where person2vec will look when it tries to find text from which to create the embeddings.

To set up a mongodb database, follow these instructions: https://docs.mongodb.com/manual/installation/

Once you have mongodb installed, make sure it is running. To make sure it's running type this into your command prompt:

```sudo service mongod start```

Once you have mongodb working, you can load all the information you want about the entities you care about into it, making sure to store any text you want to embed in a field called 'texts' as an array of strings. For an example of how to do this, you can look in the file "yelp_loader.py" in the person2vec/utils directory of this repo.

Next, you'll want to install person2vec.  To do that, clone this repo, then navigate to the directory you cloned it into.

To install necessary dependencies, you'll need to execute the command:
```pip install -e .```

If that doesn't work, you may need to do:
```sudo pip install -e .```

Once dependencies finish installing, you can begin training your embeddings. There are a few ways to do this. The easiest way is to use the command line tool. From the person2vec directory of this repo, simply type the command:

```python embed_database.py 'name_of_your_database'```

The process may take a while, depending on the size of your database, so give it time. 

If you want a bit more control, you can use the python package to play with the various settings.  To train embeddings in your own python script, the essential pieces you'll need are as follows:

```
from person2vec import data_handler
from person2vec.generators import training_data_generator
from person2vec.utils import snippet_creator
from person2vec.train_embeddings import train

# creates an object that handles transactions with the mongodb database
handler = data_handler.DataHandler('name_of_your_database')
# creates snippets from the text in the database that'll serve as training data for the embeddings
snippet_creator.snippetize_db(handler)
# a generator that, when invoked, will continuously generate training data from the snippets
data_gen = training_data_generator.EmbeddingDataGenerator(db_name=db_name)
# this command trains the embeddings and yields the trained model. 
# You can also pass train.train_model() a pre-defined or even pre-trained model, and it'll train that instead of the default 
model, data_gen = train.train_model(data_gen=data_gen)
# this saves the embeddings to the database for later use
handler.save_embeddings_to_db(model, data_gen)
```
Training of embeddings for more than 10,000 entities is not recommended in this release. This limit will improve in future releases. 

