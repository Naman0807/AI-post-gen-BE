- checks the Nginx error logs.
```bash
sudo less /var/log/nginx/error.log
```

- checks the Nginx access logs.
```bash
sudo less /var/log/nginx/access.log
```

- checks the Nginx process logs.
```bash
sudo journalctl -u nginx
```

- checks your Flask app’s Gunicorn logs.
```bash
sudo journalctl -u app
```
