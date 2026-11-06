FROM node:22-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# PORT=3001 comes from .env; the browser connects to the backend at ws://localhost:8765
# (set REACT_APP_WS_URL to point it elsewhere).
EXPOSE 3001

CMD ["npm", "start"]
