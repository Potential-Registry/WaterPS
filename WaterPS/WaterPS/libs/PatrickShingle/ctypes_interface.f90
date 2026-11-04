subroutine waterps_potential(coords, ngeom, energy) bind(C, name='waterps_potential')
    use iso_c_binding
    implicit none
    integer(c_int), value, intent(in) :: ngeom
    real(c_double), intent(in) :: coords(3,3,ngeom)
    real(c_double), intent(out) :: energy(ngeom)
    real(c_double) :: legacy_coords(1,3,3)
    integer :: i
    do i=1,ngeom
        legacy_coords(1,:,:) = transpose(coords(:,:,i))
        call calc_hoh_pot(legacy_coords, 1, energy(i))
    enddo
end subroutine
