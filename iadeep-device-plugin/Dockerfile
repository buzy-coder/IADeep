FROM golang:1.17.8-stretch as build

WORKDIR /go/src/gpushare-device-plugin
COPY . .

# RUN go mod init gpushare-device-plugin
RUN go mod vendor
RUN go mod tidy

RUN export CGO_LDFLAGS_ALLOW='-Wl,--unresolved-symbols=ignore-in-object-files' && \
    go build -ldflags="-s -w" -o /go/bin/gpushare-device-plugin cmd/nvidia/main.go

RUN go build -o /go/bin/kubectl-inspect-gpushare cmd/inspect/*.go

FROM debian:bullseye-slim

ENV NVIDIA_VISIBLE_DEVICES=ALL
ENV NVIDIA_DRIVER_CAPABILITIES=utility

COPY --from=build /go/bin/gpushare-device-plugin /usr/bin/gpushare-device-plugin

COPY --from=build /go/bin/kubectl-inspect-gpushare /usr/bin/kubectl-inspect-gpushare

CMD ["gpushare-device-plugin","-logtostderr"]
