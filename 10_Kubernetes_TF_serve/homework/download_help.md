### How to download specific directory from main repository

```
mkdir homeworks
cd homeworks

git clone --depth 1 --filter=blob:none --no-checkout https://github.com/alexeygrigorev/mlbookcamp-code
cd mlbookcamp-code
git sparse-checkout set course-zoomcamp/cohorts/2022/05-deployment/homework
```

```
cd course-zoomcamp/cohorts/2022/05-deployment/homework
```
You will find homework python, dockerfile and pipfiles files reside there. 
We will use `echo $pwd` to shorten directory source and destination when moving files 

```
cd .. (until you reache directory `cohorts`)
export source_dir=$(pwd)
mv $source_dir/2022/05-deployment/homework/ $source_dir
```
Repeat this step

```
cd .. (until you reache directory `mlbookcamp-code`)
export source_dir=$(pwd)
mv $source_dir/course-zoomcamp/cohorts/homework $source_dir
```

Then files are then one level closer to the directory created earlier.