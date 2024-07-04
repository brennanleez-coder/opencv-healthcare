Usage:
```bash
cd server/app
fastapi dev server/app/main.py
```

accesss `http://localhost:8000/docs` to see the API documentation.


Run tests:
```bash
python -m unittest server.app.tests.test_helper_fns

```

Containerise
navigate to the root of the project and run the following commands:
```bash
cd server
```

```bash
docker build -t empower-vision-be .
docker run -d --name empower-vision-be-container -p 80:80 empower-vision-be
```
