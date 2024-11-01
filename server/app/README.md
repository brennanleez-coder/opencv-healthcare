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
docker build -t frailty-vision-be .
docker run -d -p 8000:8000 frailty-vision-be

```