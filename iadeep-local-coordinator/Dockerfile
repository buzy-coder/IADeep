FROM golang:1.17.8-stretch as builder

WORKDIR /go/src/iadeep_local_coordinator
COPY . .
# RUN go mod vendor
RUN go build -o /usr/bin/iadeep_local_coordinator cmd/*.go

FROM ubuntu  

COPY --from=builder /usr/bin/iadeep_local_coordinator /usr/bin/iadeep_local_coordinator
RUN mkdir /csv
ADD csv /csv
RUN mkdir /etcd_key
ADD etcd_key /etcd_key
ENV PORT=40001
ENV THREADNESS=10
ENV ETCD_SERVER_IP="10.119.46.42"
ENV ETCD_PORT=2379