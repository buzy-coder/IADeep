# download libraries
go mod init gpushare-device-plugin
go mod vendor
go mod tidy

# build image
docker build -t 10.119.46.41:30003/library/gpushare-device-plugin:1.0 .
docker push 10.119.46.41:30003/library/gpushare-device-plugin:1.0

