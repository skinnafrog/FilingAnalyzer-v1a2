# Use nginx to serve static files
FROM nginx:alpine

# Copy the HTML files
COPY index.html /usr/share/nginx/html/
COPY filings.html /usr/share/nginx/html/

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]