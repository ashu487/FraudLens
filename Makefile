up:
	docker compose up --build
down:
	docker compose down
test:
	pytest -q
logs:
	docker compose logs -f scoring simulator
