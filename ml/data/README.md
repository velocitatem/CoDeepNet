# Data

Some thoughts on processing data: In a lot of cases you will get data not in an s3 bucket or anything glamarous and for doing anything in terms of modelling you need the data locally but then when you maybe have a 1TB dataset you want 10GB locally and then you upload to a GPU rich server and there you will want all of teh data. How can you managed this data well? What are best practices?

Huggingface lets you upload up to 300 gigs of data into a dataset.
