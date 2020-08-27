Future Releases
===============

.. toctree::
   :maxdepth: 1
   :caption: Notes
   :name: Notes

In the future, we will focus on improving each simulation's thread performance,
in the current release, any P2P network with more than 16 peers and many files
can take a long time to complete due to the amount of *for* statements that
exist in the code. This is one reason why our program offers the possibility
of running multiple (different) simulations in different threads, allowing
researchers to complete more simulations, in less time, by fully utilizing the
CPU of their machines. Ideally, we would like to offer this speed up using
multi-threading and fast individual threads.
