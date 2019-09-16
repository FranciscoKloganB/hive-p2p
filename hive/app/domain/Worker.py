import psutil

class Worker:
    """
    Defines a node on the P2P network. Workers are subject to constraints imposed by Hiveminds, constraints they inflict
    on themselves based on available computing power (CPU, RAM, etc...) and can have [0, N] shared file parts. Workers
    have the ability to reconstruct lost file parts whene needed.
    """