      program tab
      implicit none
      integer i, n, is
      real e, s, st, emin, emax, x
      real stot(3)
      integer nzs(3), nes(3), j
      data nzs /1, 6, 8/
      data nes /1, 6, 8/
      n = 5000
      emin = 5.0
      emax = 1000.0
      open(unit=10, file='verner_sigma.dat')
      do i = 0, n
         x = log10(emin) + (log10(emax)-log10(emin))*real(i)/real(n)
         e = 10.0**x
         do j = 1, 3
            st = 0.0
            do is = 1, 7
               call phfit2(nzs(j), nes(j), is, e, s)
               st = st + s
            end do
            stot(j) = st
         end do
         write(10,'(4es14.6)') e, stot(1), stot(2), stot(3)
      end do
      close(10)
      end
