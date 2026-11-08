# Stage 1: the production bundle.
FROM node:22-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# The browser, not the container, opens the WebSocket. Create React App inlines REACT_APP_*
# variables at build time, so the backend URL is a build argument (default ws://localhost:8765).
ARG REACT_APP_WS_URL=ws://localhost:8765
ENV REACT_APP_WS_URL=$REACT_APP_WS_URL \
    GENERATE_SOURCEMAP=false
RUN npm run build

# Stage 2: static files served by nginx on port 3001, the development server's port.
FROM nginx:1.28-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/build /usr/share/nginx/html

EXPOSE 3001
