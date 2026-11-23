/** One colour per noise profile of an adaptive_gate, shared by the pipeline editor and the
 *  Live page (profile i of the filter's `profiles` list). */
export const PROFILE_COLORS = ['#5aa9ff', '#ffa94d', '#4cc38a', '#c77dff', '#ef6461', '#e5b44a', '#6fd3d6', '#d0d0d0']
export const profileColor = (i: number) => PROFILE_COLORS[i % PROFILE_COLORS.length]
