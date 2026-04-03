# 🚀 End-to-End ML Deployment on AWS (SageMaker → Docker → ECR → EKS)

This document explains the **complete process** we followed to build and deploy a machine learning model as a live API on AWS.

---

# 🧠 Architecture Overview

```
SageMaker → Train Model → Store in S3 → Build API → Docker → ECR → EKS → LoadBalancer → Public API
```

---

# 🧩 Phase 1 — SageMaker Setup & Training

## Step 1: Create SageMaker Notebook

* Go to AWS Console → SageMaker
* Create Notebook Instance
* Open Jupyter Notebook

---

## Step 2: Training Script (`train.py`)

```python
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def main():
    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

    train_ds = dataset["train"].shuffle(seed=42).select(range(2000))
    eval_ds = dataset["validation"].shuffle(seed=42).select(range(500))

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    keep_cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=keep_cols)
    eval_ds.set_format(type="torch", columns=keep_cols)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    args = TrainingArguments(
        output_dir="/opt/ml/output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("/opt/ml/model")
    tokenizer.save_pretrained("/opt/ml/model")


if __name__ == "__main__":
    main()
```

---

## Step 3: Start Training Job

```python
from sagemaker.huggingface import HuggingFace
import sagemaker

role = sagemaker.get_execution_role()
sess = sagemaker.Session()
bucket = sess.default_bucket()

estimator = HuggingFace(
    entry_point="train.py",
    source_dir=".",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    transformers_version="4.12",
    pytorch_version="1.9",
    py_version="py38",
    output_path=f"s3://{bucket}/sentiment-output/"
)

estimator.fit()
```

---

## Result

* Model saved in **S3**
* File: `model.tar.gz`

---

# 🧩 Phase 2 — Build API

## `app.py`

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

classifier = pipeline("sentiment-analysis", model="./model", tokenizer="./model")

class Request(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: Request):
    result = classifier(req.text)
    return result[0]
```

---

## `requirements.txt`

```
fastapi
uvicorn
transformers
torch
pydantic
```

---

# 🧩 Phase 3 — Dockerize

## `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model ./model

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## Build Image

```bash
docker build -t sentiment-api .
```

---

# 🧩 Phase 4 — Push to ECR

```bash
aws ecr create-repository --repository-name sentiment-api

aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-2.amazonaws.com


docker tag sentiment-api:latest <account>.dkr.ecr.us-east-2.amazonaws.com/sentiment-api:latest

docker push <account>.dkr.ecr.us-east-2.amazonaws.com/sentiment-api:latest
```

---

# 🧩 Phase 5 — EKS Cluster (without Auto Mode)

* Create cluster
* Add node group (t3.medium)
* Ensure IAM role has:

  * AmazonEKSWorkerNodePolicy
  * AmazonEKS_CNI_Policy
  * AmazonEC2ContainerRegistryReadOnly

---

## Connect Cluster

```bash
aws eks update-kubeconfig --region us-east-2 --name model
kubectl get nodes
```

---

# 🧩 Phase 6 — Kubernetes Deployment

## `k8s.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: sentiment-api
  template:
    metadata:
      labels:
        app: sentiment-api
    spec:
      containers:
        - name: api
          image: <account>.dkr.ecr.us-east-2.amazonaws.com/sentiment-api:latest
          ports:
            - containerPort: 8080
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "1"
              memory: "2Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: sentiment-api
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-scheme: internet-facing
spec:
  type: LoadBalancer
  selector:
    app: sentiment-api
  ports:
    - port: 80
      targetPort: 8080
```

---

## Deploy

```bash
kubectl apply -f k8s.yaml
kubectl get pods
kubectl get svc sentiment-api
```

---

# 🧪 Phase 7 — Test API

```bash
curl http://<ELB>/health
```

```bash
curl -X POST http://<ELB>/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I love this product"}'
```

---

# 🎉 Final Result

* ML model trained ✅
* Stored in S3 ✅
* API created ✅
* Dockerized ✅
* Pushed to ECR ✅
* Deployed on EKS ✅
* Public endpoint available ✅

---

# 💰 Cleanup

```bash
kubectl delete -f k8s.yaml
```

Delete cluster from AWS console.

---

# 🚀 Outcome

You built a **production-ready ML API on AWS** 🚀
