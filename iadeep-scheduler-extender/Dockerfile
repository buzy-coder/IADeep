FROM golang:1.17.8-stretch as build

WORKDIR /go/src/gpushare-scheduler-extender
COPY . .

# RUN go mod init iadeep-scheduler-extender
RUN go mod tidy
RUN go mod vendor
RUN go build -o /go/bin/gpushare-sche-extender cmd/*.go

FROM ubuntu

RUN mkdir /csv
ADD csv /csv
RUN mkdir /etcd_key
ADD etcd_key /etcd_key

COPY --from=build /go/bin/gpushare-sche-extender /usr/bin/gpushare-sche-extender
# scheduler: IADEEP, ANTMAN, KERNELEST
ENV SCHEDULER=KERNELEST
ENV ETCD_SERVER_IP="10.119.46.42"
ENV ETCD_PORT=2379
ENV ENABLE_OPTIMIZER=true
ENV ENABLE_PREDICOR=true
ENV ENABLE_TUNING=true
CMD ["gpushare-sche-extender"]
