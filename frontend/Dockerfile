FROM node:20-alpine AS builder

WORKDIR /app

ARG VITE_API_URL=http://localhost:8000
ARG VITE_WS_URL=ws://localhost:8000
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_WS_URL=$VITE_WS_URL

COPY package*.json ./
# Cache npm entre les rebuilds (BuildKit)
RUN --mount=type=cache,target=/root/.npm \
    npm ci --prefer-offline

COPY . .
RUN npm run build

# --- Serve avec nginx ---
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
