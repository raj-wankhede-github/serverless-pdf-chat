import os, json
import boto3
from aws_lambda_powertools import Logger
from langchain.embeddings import BedrockEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS


DOCUMENT_TABLE = os.environ["DOCUMENT_TABLE"]
BUCKET = os.environ["BUCKET"]

s3 = boto3.client("s3")
ddb = boto3.resource("dynamodb")
document_table = ddb.Table(DOCUMENT_TABLE)
logger = Logger()


def set_doc_status(user_id, document_id, status):
    print("setting doc status")
    document_table.update_item(
        Key={"userid": user_id, "documentid": document_id},
        UpdateExpression="SET docstatus = :docstatus",
        ExpressionAttributeValues={":docstatus": status},
    )


@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    print(event)
    event_body = json.loads(event["Records"][0]["body"])
    document_id = event_body["documentid"]
    user_id = event_body["user"]
    key = event_body["key"]
    file_name_full = key.split("/")[-1]
    
    print(document_id)
    print(user_id)
    print(key)
    print(file_name_full)

    set_doc_status(user_id, document_id, "PROCESSING")

    print("S3 downloading file")
    s3.download_file(BUCKET, key, f"/tmp/{file_name_full}")
    print("S3 file downlaoded")

    print("laoding via pyPDFLoader")
    loader = PyPDFLoader(f"/tmp/{file_name_full}")
    
    print("loaded using pyPDFLoader")

    print("boto3 client for bedrock-runtime")
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
    )

    print("Bedrock embeddings")
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        client=bedrock_runtime,
        region_name="us-east-1",
    )

    print("Vector store index creator")
    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
    )

    print("Index creater from Loaders")
    index_from_loader = index_creator.from_loaders([loader])

    print("Save local - Vector store")
    index_from_loader.vectorstore.save_local("/tmp")

    print("Uploading files to S3")
    s3.upload_file(
        "/tmp/index.faiss", BUCKET, f"{user_id}/{file_name_full}/index.faiss"
    )
    s3.upload_file("/tmp/index.pkl", BUCKET, f"{user_id}/{file_name_full}/index.pkl")

    set_doc_status(user_id, document_id, "READY")
