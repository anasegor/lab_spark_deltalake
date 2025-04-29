Run:
```
docker build -t my-spark-app .
docker-compose run spark-app /app/run.sh
```
Logs :

```
cat ./logs/output.log
```

MLflow logs are in ./logs/mlruns