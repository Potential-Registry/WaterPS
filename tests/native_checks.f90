program native_checks
    use iso_c_binding
    use, intrinsic :: ieee_arithmetic
    implicit none
    interface
        subroutine evaluate(x,n,v) bind(C,name='waterps_potential')
            import c_int,c_double
            integer(c_int), value :: n
            real(c_double), intent(in) :: x(3,3,n)
            real(c_double), intent(out) :: v(n)
        end subroutine
    end interface
    real(c_double) :: x(3,3,1),base(3,3,1),v(1)
    integer :: i,j,k
    integer(c_int) :: status(1)
    call ieee_set_halting_mode(ieee_invalid,.true.)
    call ieee_set_halting_mode(ieee_divide_by_zero,.true.)
    call ieee_set_halting_mode(ieee_overflow,.true.)
    base=reshape([ &
        1.80138030982117514d+00, -2.84874506400843615d-02, 6.88207232460313120d-03, &
        -4.71329257482190567d-01, 1.79190301891723935d+00, 1.70569485856851555d-02, &
        5.16405004668164108d-02, -9.16869733696078215d-03, 1.62388327385106813d-02 ],[3,3,1])
    do i=1,32
        x=base
        do j=1,size(x,2)
            do k=1,3
                x(k,j,1)=x(k,j,1)+0.002d0*cos(dble(i*j*k))
            enddo
        enddo
        call evaluate(x,1,v)
        if (.not.all(ieee_is_finite(v))) error stop "nonfinite energy"
    enddo
    print *, '32 evaluations passed IEEE and bounds checks'
end program
