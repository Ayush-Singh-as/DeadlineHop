from dataclasses import dataclass


@dataclass(frozen=True)
class Packet:
    id: int
    arrival: int
    duration: int

    @property
    def deadline(self):
        return self.arrival + self.duration - 1


def edf_key(packet):
    return packet.deadline, packet.arrival, packet.id

