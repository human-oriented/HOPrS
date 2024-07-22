import * as Icon from 'react-feather';

export const LoadingButton = (props) => {
    const {loading, width} = props
    return(
        <button className="bg-black hover:opacity-90 text-white font-bold px-10 py-4 rounded-tl-2xl rounded-br-2xl  min-w-[300px] mr-2">
            <div className='flex flex-row items-center justify-center'>
            {props.children}
            {loading && 
                <Icon.Loader className='ml-2 animate-spin' />
            }
            </div>
        </button>
    )
}