# API rate limits

The Lumen API allows 100 requests per minute per token on the Starter plan and 1,000 requests per minute on the Business plan.
Requests over the limit receive HTTP 429 with a Retry-After header in seconds.
Bulk ingestion endpoints have a separate limit of 50 MB per request.
Clients should retry with exponential backoff and must not retry immediately in a tight loop.
